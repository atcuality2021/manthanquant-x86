"""
TurboQuant 3-bit Lloyd-Max Vector Quantizer — Pure PyTorch implementation.

This is the Phase 1 (correctness-first) encoder/decoder. All operations are
vectorized PyTorch ops on GPU. No Python for-loops over vectors.

Algorithm (Zandieh et al., "TurboQuant", 2025):
  ENCODE:
    1. r = ||x||_2                          (radius, preserves magnitude)
    2. x_hat = x / r                        (unit sphere)
    3. x_scaled = x_hat * sqrt(D)           (coordinates ~ N(0,1) by CLT)
    4. idx = searchsorted(boundaries, x_scaled)  (nearest Lloyd-Max centroid)
    5. packed = bit_pack(idx, 3 bits)        (into int32 words)
    Store: r (float32) + packed (int32[ceil(D*3/32)])

  DECODE:
    1. idx = bit_unpack(packed, 3 bits)
    2. x_scaled = centroids[idx]
    3. x_hat = x_scaled / sqrt(D)
    4. x_reconstructed = x_hat * r

Compression ratio for D=128, bf16:
  Original:   128 * 2 = 256 bytes
  Compressed: 4 + ceil(128*3/32)*4 = 4 + 48 = 52 bytes
  Ratio:      256 / 52 = 4.92x

For D=256, bf16:
  Original:   256 * 2 = 512 bytes
  Compressed: 4 + ceil(256*3/32)*4 = 4 + 96 = 100 bytes
  Ratio:      512 / 100 = 5.12x
"""

import torch
from typing import Optional

# Lloyd-Max optimal centroids for 3-bit (8 levels) quantization of N(0,1).
# Computed by solving the continuous 1D k-means problem on the standard
# normal distribution using iterative EM (Lloyd-Max algorithm).
# MSE = 0.03455 per coordinate (D_mse approx 0.03 for b=3).
CENTROIDS_3BIT = torch.tensor([
    -2.151946, -1.343910, -0.756006, -0.245094,
     0.245094,  0.756006,  1.343910,  2.151946
], dtype=torch.float32)

# Decision boundaries = midpoints between consecutive centroids.
BOUNDARIES_3BIT = torch.tensor([
    -1.747928, -1.049958, -0.500550, 0.000000,
     0.500550,  1.049958,  1.747928
], dtype=torch.float32)

# Precomputed for 2-bit and 4-bit (for future mixed-precision support)
CENTROIDS_2BIT = torch.tensor([
    -1.510234, -0.453198, 0.453198, 1.510234
], dtype=torch.float32)

BOUNDARIES_2BIT = torch.tensor([
    -0.981716, 0.000000, 0.981716
], dtype=torch.float32)

CENTROIDS_4BIT = torch.tensor([
    -2.733075, -2.069016, -1.618422, -1.256206,
    -0.942057, -0.656532, -0.388180, -0.127961,
     0.127961,  0.388180,  0.656532,  0.942057,
     1.256206,  1.618422,  2.069016,  2.733075
], dtype=torch.float32)

BOUNDARIES_4BIT = torch.tensor([
    -2.401046, -1.843719, -1.437314, -1.099132,
    -0.799295, -0.522356, -0.258071, 0.000000,
     0.258071,  0.522356,  0.799295,  1.099132,
     1.437314,  1.843719,  2.401046
], dtype=torch.float32)

CODEBOOKS = {
    2: (CENTROIDS_2BIT, BOUNDARIES_2BIT),
    3: (CENTROIDS_3BIT, BOUNDARIES_3BIT),
    4: (CENTROIDS_4BIT, BOUNDARIES_4BIT),
}


# ── Bit-pack index tables (cached per D, bits, device) ──────────────────
#
# When bits=3 and D=128, some indices straddle int32 word boundaries.
# E.g. coordinate 10: bit_pos=30, needs bits [30,31,32] but int32 only has
# bits [0,31]. We handle this by splitting boundary-crossing values into
# lower and upper parts across two words.

_pack_tables: dict[tuple, tuple] = {}


def _get_pack_tables(
    D: int, bits: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute vectorized pack/unpack index tables.

    Returns:
        word_idx:    [D] — primary word index for each coordinate
        bit_offset:  [D] — bit offset within primary word
        overflow_mask: [D] — bool, True if value crosses word boundary
        overflow_word: [D] — secondary word index (word_idx + 1)
        overflow_shift: [D] — how many bits go into the primary word
    """
    key = (D, bits, str(device))
    if key not in _pack_tables:
        pos = torch.arange(D, device=device)
        bit_pos = pos * bits
        word_idx = bit_pos // 32
        bit_offset = bit_pos % 32

        # Detect boundary crossings: value needs bits beyond bit 31
        overflow_mask = (bit_offset + bits) > 32
        overflow_word = word_idx + 1
        # Bits that fit in the primary word
        overflow_shift = 32 - bit_offset  # bits in lower word

        _pack_tables[key] = (word_idx, bit_offset, overflow_mask, overflow_word, overflow_shift)
    return _pack_tables[key]


# ── Encoder ──────────────────────────────────────────────────────────────


class TurboQuantEncoder:
    """Stateless 3-bit Lloyd-Max encoder.

    All state is in the codebook constants. Thread-safe, no side effects.
    """

    def __init__(self, bits: int = 3):
        assert bits in CODEBOOKS, f"Supported bit-widths: {list(CODEBOOKS.keys())}"
        self.bits = bits
        self.centroids, self.boundaries = CODEBOOKS[bits]

    def encode(
        self, vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode vectors to compressed representation.

        Args:
            vectors: [N, D] tensor on any device, any float dtype.

        Returns:
            radii:  [N] float32 tensor — L2 norms
            packed: [N, num_words] int32 tensor — bit-packed centroid indices
                    where num_words = ceil(D * bits / 32)
        """
        assert vectors.ndim == 2, f"Expected [N, D], got shape {vectors.shape}"
        N, D = vectors.shape
        device = vectors.device

        # Move codebook to device (cached by PyTorch)
        boundaries = self.boundaries.to(device)

        # 1. Float32 for numerical stability
        x = vectors.float()

        # 2. Compute L2 radius
        radii = torch.norm(x, dim=1)  # [N]

        # 3. Normalize to unit sphere, scale to N(0,1)
        safe_radii = radii.clamp(min=1e-8)
        x_scaled = (x / safe_radii.unsqueeze(1)) * (D ** 0.5)  # [N, D]

        # 4. Quantize: find nearest centroid index via boundary search
        # searchsorted expects sorted 1D tensor, returns index in [0, len(boundaries)]
        indices = torch.searchsorted(boundaries, x_scaled.reshape(-1))
        indices = indices.reshape(N, D).to(torch.int32)  # [N, D], values in [0, 2^bits-1]

        # 5. Bit-pack into int32 words (handling word boundary crossings)
        num_words = (D * self.bits + 31) // 32
        word_idx, bit_offset, overflow_mask, overflow_word, overflow_shift = \
            _get_pack_tables(D, self.bits, device)

        mask = (1 << self.bits) - 1
        idx_masked = (indices & mask).to(torch.int64)  # [N, D]

        # Primary part: lower bits that fit in the primary word
        shifted = idx_masked << bit_offset.unsqueeze(0)  # [N, D]

        # For overflow coordinates, mask to only the bits that fit
        if overflow_mask.any():
            # Where overflow happens, only keep lower `overflow_shift` bits
            lower_mask = ((1 << overflow_shift) - 1).unsqueeze(0).to(torch.int64)  # [1, D]
            shifted_lower = (idx_masked & lower_mask) << bit_offset.unsqueeze(0)
            # Upper bits go to next word, shifted to bit 0
            shifted_upper = idx_masked >> overflow_shift.unsqueeze(0)

            # Use the non-overflow shifted for normal coords, lower for overflow
            of = overflow_mask.unsqueeze(0)  # [1, D]
            shifted = torch.where(of, shifted_lower, shifted)
        else:
            shifted_upper = None

        # Scatter-add primary parts
        word_idx_exp = word_idx.unsqueeze(0).expand(N, -1).long()
        packed = torch.zeros(N, num_words, dtype=torch.int64, device=device)
        packed.scatter_add_(1, word_idx_exp, shifted)

        # Scatter-add overflow parts
        if shifted_upper is not None and overflow_mask.any():
            # Only add overflow values for coordinates that actually overflow
            overflow_vals = torch.where(
                overflow_mask.unsqueeze(0), shifted_upper,
                torch.zeros_like(shifted_upper)
            )
            # Clamp overflow_word to valid range
            ow_clamped = overflow_word.clamp(max=num_words - 1)
            ow_exp = ow_clamped.unsqueeze(0).expand(N, -1).long()
            packed.scatter_add_(1, ow_exp, overflow_vals)

        packed = packed.to(torch.int32)
        return radii, packed


class TurboQuantDecoder:
    """Stateless 3-bit Lloyd-Max decoder."""

    def __init__(self, bits: int = 3):
        assert bits in CODEBOOKS
        self.bits = bits
        self.centroids, _ = CODEBOOKS[bits]

    def decode(
        self,
        radii: torch.Tensor,
        packed: torch.Tensor,
        D: int,
    ) -> torch.Tensor:
        """Decode compressed representation back to vectors.

        Args:
            radii:  [N] float32 tensor — L2 norms
            packed: [N, num_words] int32 tensor — bit-packed indices
            D:      original vector dimension

        Returns:
            vectors: [N, D] float32 tensor — reconstructed vectors
        """
        N = radii.shape[0]
        device = radii.device
        centroids = self.centroids.to(device)
        idx_mask = (1 << self.bits) - 1

        word_idx, bit_offset, overflow_mask, overflow_word, overflow_shift = \
            _get_pack_tables(D, self.bits, device)

        # Work in int64 to avoid sign issues
        packed_i64 = packed.to(torch.int64)

        # Gather primary word for each coordinate
        word_idx_exp = word_idx.unsqueeze(0).expand(N, -1).long()
        words = torch.gather(packed_i64, 1, word_idx_exp)  # [N, D]

        # Extract lower bits from primary word
        indices = (words >> bit_offset.unsqueeze(0)) & idx_mask  # [N, D]

        # Handle overflow: gather upper bits from next word
        if overflow_mask.any():
            ow_clamped = overflow_word.clamp(max=packed.shape[1] - 1)
            ow_exp = ow_clamped.unsqueeze(0).expand(N, -1).long()
            overflow_words = torch.gather(packed_i64, 1, ow_exp)  # [N, D]

            # Upper bits: extract from bit 0 of overflow word
            upper_bits_count = (self.bits - overflow_shift).clamp(min=0)
            upper_mask = ((1 << upper_bits_count) - 1).unsqueeze(0).to(torch.int64)
            upper_bits = (overflow_words & upper_mask) << overflow_shift.unsqueeze(0)

            # Combine: lower bits from primary word + upper bits from overflow word
            # Only apply to coordinates that actually overflow
            indices = torch.where(
                overflow_mask.unsqueeze(0),
                ((words >> bit_offset.unsqueeze(0)) & ((1 << overflow_shift) - 1).unsqueeze(0).to(torch.int64)) | upper_bits,
                indices,
            )
            indices = indices & idx_mask  # Final mask to ensure valid range

        # Look up centroids
        x_scaled = centroids[indices.long()]  # [N, D]

        # Undo scaling and restore magnitude
        safe_radii = radii.clamp(min=1e-8)
        reconstructed = (x_scaled / (D ** 0.5)) * safe_radii.unsqueeze(1)

        return reconstructed


# ── CUDA kernel backend (Phase 2) ────────────────────────────────────────

_C = None
HAS_CUDA_KERNELS = False

try:
    import manthanquant._C as _C
    HAS_CUDA_KERNELS = True
except ImportError:
    pass


# ── Convenience functions ────────────────────────────────────────────────


_default_encoder = TurboQuantEncoder(bits=3)
_default_decoder = TurboQuantDecoder(bits=3)


def encode(
    vectors: torch.Tensor, bits: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode vectors using 3-bit Lloyd-Max quantization.

    Auto-selects CUDA kernels (if built) or PyTorch fallback.

    Args:
        vectors: [N, D] tensor on any device
        bits: quantization bits (2, 3, or 4)

    Returns:
        (radii, packed) — both on same device as input
    """
    # Use CUDA kernels if available and input is on GPU
    if HAS_CUDA_KERNELS and vectors.is_cuda:
        return _C.tq_encode(vectors.float().contiguous(), 42, bits)

    if bits == 3:
        return _default_encoder.encode(vectors)
    return TurboQuantEncoder(bits).encode(vectors)


def decode(
    radii: torch.Tensor, packed: torch.Tensor, D: int, bits: int = 3
) -> torch.Tensor:
    """Decode compressed vectors.

    Auto-selects CUDA kernels (if built) or PyTorch fallback.

    Args:
        radii: [N] float32 norms
        packed: [N, words] int32 bit-packed indices
        D: original dimension
        bits: quantization bits

    Returns:
        [N, D] float32 reconstructed vectors
    """
    if HAS_CUDA_KERNELS and radii.is_cuda:
        return _C.tq_decode(radii, packed, D, 42, bits)

    if bits == 3:
        return _default_decoder.decode(radii, packed, D)
    return TurboQuantDecoder(bits).decode(radii, packed, D)


def compression_ratio(D: int, bits: int = 3, dtype_bytes: int = 2) -> float:
    """Calculate compression ratio for given dimension and dtype.

    Args:
        D: vector dimension (e.g. 128 for head_dim)
        bits: quantization bits
        dtype_bytes: original dtype size (2 for bf16/fp16, 4 for fp32)

    Returns:
        compression ratio (e.g. 5.12 for D=256, bits=3, bf16)
    """
    original = D * dtype_bytes
    num_words = (D * bits + 31) // 32
    compressed = 4 + num_words * 4  # radius (f32) + packed (int32 words)
    return original / compressed
