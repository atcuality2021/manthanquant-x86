"""
Tests for TurboQuant encoder/decoder correctness.

Run: python -m pytest manthanquant/tests/test_quantizer.py -v
"""

import torch
import math
import pytest

from manthanquant.core.quantizer import (
    TurboQuantEncoder,
    TurboQuantDecoder,
    encode,
    decode,
    compression_ratio,
    CENTROIDS_3BIT,
    BOUNDARIES_3BIT,
)


# ── Test fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def encoder():
    return TurboQuantEncoder(bits=3)


@pytest.fixture
def decoder():
    return TurboQuantDecoder(bits=3)


def random_kv_vectors(N, D, device, dtype=torch.bfloat16):
    """Generate random vectors simulating KV cache data."""
    return torch.randn(N, D, dtype=dtype, device=device)


# ── Correctness tests ────────────────────────────────────────────────────

class TestEncodeDecode:
    """Test that encode→decode roundtrip preserves vector quality."""

    def test_basic_roundtrip(self, encoder, decoder, device):
        """Encode then decode should give vectors close to original."""
        vectors = random_kv_vectors(64, 128, device)
        radii, packed = encoder.encode(vectors)
        reconstructed = decoder.decode(radii, packed, D=128)

        # Cosine similarity should be high
        cos_sim = torch.nn.functional.cosine_similarity(
            vectors.float(), reconstructed, dim=1
        )
        assert cos_sim.mean() > 0.95, f"Mean cosine similarity {cos_sim.mean():.4f} < 0.95"

    def test_various_dimensions(self, encoder, decoder, device):
        """Test with different head dimensions (64, 128, 256)."""
        for D in [64, 128, 256]:
            vectors = random_kv_vectors(32, D, device)
            radii, packed = encoder.encode(vectors)
            reconstructed = decoder.decode(radii, packed, D=D)

            cos_sim = torch.nn.functional.cosine_similarity(
                vectors.float(), reconstructed, dim=1
            )
            assert cos_sim.mean() > 0.93, (
                f"D={D}: cosine {cos_sim.mean():.4f} < 0.93"
            )

    def test_radius_preservation(self, encoder, decoder, device):
        """Encoded radii should match original L2 norms."""
        vectors = random_kv_vectors(100, 128, device)
        radii, packed = encoder.encode(vectors)

        original_norms = torch.norm(vectors.float(), dim=1)
        torch.testing.assert_close(
            radii, original_norms, rtol=1e-5, atol=1e-6
        )

    def test_zero_vectors(self, encoder, decoder, device):
        """Zero vectors should not crash (clamped radius)."""
        vectors = torch.zeros(4, 128, device=device)
        radii, packed = encoder.encode(vectors)
        reconstructed = decoder.decode(radii, packed, D=128)

        # Reconstructed should also be near-zero
        assert reconstructed.abs().max() < 0.01

    def test_large_batch(self, encoder, decoder, device):
        """Test with batch sizes typical of vLLM (thousands of vectors)."""
        # Simulates: 16 tokens * 8 kv_heads = 128 vectors per layer
        vectors = random_kv_vectors(1024, 128, device)
        radii, packed = encoder.encode(vectors)
        reconstructed = decoder.decode(radii, packed, D=128)

        cos_sim = torch.nn.functional.cosine_similarity(
            vectors.float(), reconstructed, dim=1
        )
        assert cos_sim.mean() > 0.95

    def test_output_shapes(self, encoder, device):
        """Verify output tensor shapes are correct."""
        N, D, bits = 64, 128, 3
        vectors = random_kv_vectors(N, D, device)
        radii, packed = encoder.encode(vectors)

        assert radii.shape == (N,), f"radii shape {radii.shape}"
        expected_words = (D * bits + 31) // 32  # ceil(128*3/32) = 12
        assert packed.shape == (N, expected_words), f"packed shape {packed.shape}"

    def test_output_dtypes(self, encoder, device):
        """Verify output dtypes."""
        vectors = random_kv_vectors(32, 128, device)
        radii, packed = encoder.encode(vectors)

        assert radii.dtype == torch.float32
        assert packed.dtype == torch.int32

    def test_device_consistency(self, encoder, decoder, device):
        """Output tensors should be on same device as input."""
        vectors = random_kv_vectors(32, 128, device)
        radii, packed = encoder.encode(vectors)
        reconstructed = decoder.decode(radii, packed, D=128)

        assert radii.device == vectors.device
        assert packed.device == vectors.device
        assert reconstructed.device == vectors.device

    def test_input_dtype_invariance(self, encoder, decoder, device):
        """Should work with bf16, fp16, fp32 inputs."""
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            vectors = random_kv_vectors(32, 128, device, dtype=dtype)
            radii, packed = encoder.encode(vectors)
            reconstructed = decoder.decode(radii, packed, D=128)

            cos_sim = torch.nn.functional.cosine_similarity(
                vectors.float(), reconstructed, dim=1
            )
            assert cos_sim.mean() > 0.93, (
                f"dtype={dtype}: cosine {cos_sim.mean():.4f} < 0.93"
            )


class TestBitWidths:
    """Test different quantization bit-widths."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_encode_decode(self, bits, device):
        vectors = random_kv_vectors(64, 128, device)
        radii, packed = encode(vectors, bits=bits)
        reconstructed = decode(radii, packed, D=128, bits=bits)

        cos_sim = torch.nn.functional.cosine_similarity(
            vectors.float(), reconstructed, dim=1
        )
        # Quality should improve with more bits
        min_quality = {2: 0.85, 3: 0.93, 4: 0.97}
        assert cos_sim.mean() > min_quality[bits], (
            f"bits={bits}: cosine {cos_sim.mean():.4f} < {min_quality[bits]}"
        )

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_packed_size(self, bits, device):
        """Verify packed tensor has correct number of words."""
        D = 128
        vectors = random_kv_vectors(32, D, device)
        _, packed = encode(vectors, bits=bits)

        expected_words = (D * bits + 31) // 32
        assert packed.shape[1] == expected_words


class TestCompressionRatio:
    """Test compression ratio calculations."""

    def test_d128_bf16_3bit(self):
        ratio = compression_ratio(D=128, bits=3, dtype_bytes=2)
        assert ratio > 4.5, f"D=128 ratio {ratio} should be > 4.5"

    def test_d256_bf16_3bit(self):
        ratio = compression_ratio(D=256, bits=3, dtype_bytes=2)
        assert abs(ratio - 5.12) < 0.1, f"D=256 ratio {ratio} should be ~5.12"

    def test_more_bits_less_compression(self):
        r2 = compression_ratio(D=128, bits=2)
        r3 = compression_ratio(D=128, bits=3)
        r4 = compression_ratio(D=128, bits=4)
        assert r2 > r3 > r4, "More bits should mean less compression"


class TestCompressedCache:
    """Test the CompressedKVCache two-tier system."""

    def test_compress_decompress_block(self, device):
        from manthanquant.vllm_integration.compressed_cache import CompressedKVCache

        cache = CompressedKVCache(bits=3, device=device)

        block_size = 16
        num_kv_heads = 8
        head_dim = 128

        # Simulate a full block of KV data
        key_data = torch.randn(block_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        val_data = torch.randn(block_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)

        # Compress
        cache.compress_block(
            block_id=42,
            key_cache=key_data,
            value_cache=val_data,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        assert cache.is_compressed(42)
        stats = cache.get_stats()
        assert stats["compressions"] == 1
        assert stats["compression_ratio"] > 4.0

        # Decompress
        key_out = torch.zeros_like(key_data)
        val_out = torch.zeros_like(val_data)
        success = cache.decompress_block(42, key_out, val_out)

        assert success
        assert stats["decompressions"] == 0  # stats updated after call

        # Check quality
        k_cos = torch.nn.functional.cosine_similarity(
            key_data.float().reshape(-1, head_dim),
            key_out.float().reshape(-1, head_dim),
            dim=1,
        )
        assert k_cos.mean() > 0.93, f"Key cosine {k_cos.mean():.4f}"

    def test_release_block(self, device):
        from manthanquant.vllm_integration.compressed_cache import CompressedKVCache

        cache = CompressedKVCache(bits=3, device=device)

        key_data = torch.randn(16, 8, 128, device=device, dtype=torch.bfloat16)
        val_data = torch.randn(16, 8, 128, device=device, dtype=torch.bfloat16)

        cache.compress_block(0, key_data, val_data, 16, 8, 128)
        assert cache.is_compressed(0)

        cache.release_block(0)
        assert not cache.is_compressed(0)

    def test_nonexistent_block(self, device):
        from manthanquant.vllm_integration.compressed_cache import CompressedKVCache

        cache = CompressedKVCache(bits=3, device=device)
        key_out = torch.zeros(16, 8, 128, device=device)
        val_out = torch.zeros(16, 8, 128, device=device)

        assert not cache.decompress_block(999, key_out, val_out)


class TestCodebook:
    """Verify Lloyd-Max codebook properties."""

    def test_centroids_sorted(self):
        """Centroids must be in ascending order."""
        for i in range(len(CENTROIDS_3BIT) - 1):
            assert CENTROIDS_3BIT[i] < CENTROIDS_3BIT[i + 1]

    def test_boundaries_are_midpoints(self):
        """Boundaries should be midpoints between consecutive centroids."""
        for i in range(len(BOUNDARIES_3BIT)):
            expected = (CENTROIDS_3BIT[i] + CENTROIDS_3BIT[i + 1]) / 2
            assert abs(BOUNDARIES_3BIT[i] - expected) < 0.01, (
                f"Boundary {i}: {BOUNDARIES_3BIT[i]} != midpoint {expected}"
            )

    def test_centroids_symmetric(self):
        """Lloyd-Max centroids for N(0,1) should be symmetric around 0."""
        n = len(CENTROIDS_3BIT)
        for i in range(n // 2):
            assert abs(CENTROIDS_3BIT[i] + CENTROIDS_3BIT[n - 1 - i]) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
