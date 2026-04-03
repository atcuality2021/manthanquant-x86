/*
 * TurboQuant CUDA Kernels — 3-bit Lloyd-Max KV Cache Compression
 *
 * BiltIQ AI — ManthanQuant x86
 *
 * Two kernels:
 *   tq_encode: [N, D] float32 → radii [N] + packed [N, words] int32
 *   tq_decode: radii [N] + packed [N, words] → [N, D] float32
 *
 * Architecture: One CUDA block per vector, D threads cooperate.
 * Shared memory used for parallel L2 norm reduction.
 *
 * Supports SM 8.0+ (Ampere, Ada, Hopper, Blackwell)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

// ── Lloyd-Max 3-bit codebook (device constant memory) ───────────────────

// 8 centroids for 3-bit quantization of N(0,1)
__constant__ float C3[8] = {
    -2.151946f, -1.343910f, -0.756006f, -0.245094f,
     0.245094f,  0.756006f,  1.343910f,  2.151946f
};

// 7 decision boundaries (midpoints between consecutive centroids)
__constant__ float B3[7] = {
    -1.747928f, -1.049958f, -0.500550f, 0.000000f,
     0.500550f,  1.049958f,  1.747928f
};

// 4 centroids for 2-bit
__constant__ float C2[4] = {
    -1.510234f, -0.453198f, 0.453198f, 1.510234f
};
__constant__ float B2[3] = {
    -0.981716f, 0.000000f, 0.981716f
};

// 16 centroids for 4-bit
__constant__ float C4[16] = {
    -2.733075f, -2.069016f, -1.618422f, -1.256206f,
    -0.942057f, -0.656532f, -0.388180f, -0.127961f,
     0.127961f,  0.388180f,  0.656532f,  0.942057f,
     1.256206f,  1.618422f,  2.069016f,  2.733075f
};
__constant__ float B4[15] = {
    -2.401046f, -1.843719f, -1.437314f, -1.099132f,
    -0.799295f, -0.522356f, -0.258071f, 0.000000f,
     0.258071f,  0.522356f,  0.799295f,  1.099132f,
     1.437314f,  1.843719f,  2.401046f
};


// ── Encode kernel ───────────────────────────────────────────────────────

/*
 * One block per vector. Thread i handles coordinate i (and loops if D > blockDim).
 * Shared memory for parallel reduction (L2 norm).
 *
 * Steps:
 *   1. Parallel reduction to compute ||x||_2
 *   2. Normalize: x_hat = x / ||x||, scale: x_scaled = x_hat * sqrt(D)
 *   3. Quantize: binary search in boundaries → index in [0, 2^bits-1]
 *   4. Bit-pack: atomicOr indices into int32 packed words
 */
template <int BITS>
__global__ void tq_encode_kernel(
    const float* __restrict__ input,  // [N, D]
    float* __restrict__ radii,        // [N]
    int32_t* __restrict__ packed,      // [N, num_words]
    const int N,
    const int D,
    const int num_words
) {
    const int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* x = input + vec_id * D;
    int32_t* out_packed = packed + vec_id * num_words;

    extern __shared__ float sdata[];

    // ── Step 1: Parallel L2 norm ──
    float partial_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = x[i];
        partial_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_sq += __shfl_down_sync(0xFFFFFFFF, partial_sq, offset);
    }

    // Store warp results to shared memory
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) sdata[warp_id] = partial_sq;
    __syncthreads();

    // First warp reduces across warps
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        float val = (lane < num_warps) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane == 0) {
            sdata[0] = val;  // total sum of squares
        }
    }
    __syncthreads();

    float radius = sqrtf(sdata[0]);
    if (threadIdx.x == 0) {
        radii[vec_id] = radius;
    }

    float inv_radius = (radius > 1e-8f) ? (1.0f / radius) : 0.0f;
    float scale = sqrtf((float)D);

    // ── Zero output packed words ──
    for (int w = threadIdx.x; w < num_words; w += blockDim.x) {
        out_packed[w] = 0;
    }
    __syncthreads();

    // ── Steps 2-4: Normalize, quantize, bit-pack ──
    // Select boundaries based on BITS template parameter
    const float* boundaries;
    int num_boundaries;
    if constexpr (BITS == 2) {
        boundaries = B2; num_boundaries = 3;
    } else if constexpr (BITS == 3) {
        boundaries = B3; num_boundaries = 7;
    } else {
        boundaries = B4; num_boundaries = 15;
    }

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        // Normalize and scale
        float val = x[i] * inv_radius * scale;

        // Binary search for nearest centroid (searchsorted)
        int idx = 0;
        for (int b = 0; b < num_boundaries; b++) {
            if (val >= boundaries[b]) idx = b + 1;
        }

        // Bit-pack: compute word and offset
        int bit_pos = i * BITS;
        int word = bit_pos / 32;
        int offset = bit_pos % 32;

        // Primary part (always fits)
        int bits_in_word = min(BITS, 32 - offset);
        int lower_mask = (1 << bits_in_word) - 1;
        atomicOr(&out_packed[word], (idx & lower_mask) << offset);

        // Overflow part (crosses word boundary)
        if (bits_in_word < BITS && (word + 1) < num_words) {
            int upper_bits = idx >> bits_in_word;
            atomicOr(&out_packed[word + 1], upper_bits);
        }
    }
}


// ── Decode kernel ───────────────────────────────────────────────────────

template <int BITS>
__global__ void tq_decode_kernel(
    const float* __restrict__ radii,   // [N]
    const int32_t* __restrict__ packed, // [N, num_words]
    float* __restrict__ output,        // [N, D]
    const int N,
    const int D,
    const int num_words
) {
    const int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const int32_t* in_packed = packed + vec_id * num_words;
    float* out = output + vec_id * D;

    float radius = radii[vec_id];
    float inv_scale = (radius > 1e-8f) ? (radius / sqrtf((float)D)) : 0.0f;

    // Select centroids based on BITS
    const float* centroids;
    if constexpr (BITS == 2) {
        centroids = C2;
    } else if constexpr (BITS == 3) {
        centroids = C3;
    } else {
        centroids = C4;
    }
    int mask = (1 << BITS) - 1;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        int bit_pos = i * BITS;
        int word = bit_pos / 32;
        int offset = bit_pos % 32;

        // Extract index (handle word boundary crossing)
        int bits_in_word = min(BITS, 32 - offset);
        int idx = (in_packed[word] >> offset) & ((1 << bits_in_word) - 1);

        if (bits_in_word < BITS && (word + 1) < num_words) {
            int upper_bits = in_packed[word + 1] & ((1 << (BITS - bits_in_word)) - 1);
            idx |= upper_bits << bits_in_word;
        }
        idx &= mask;

        // Look up centroid and rescale
        out[i] = centroids[idx] * inv_scale;
    }
}


// ── Host-side dispatch ──────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor> tq_encode(
    torch::Tensor input,
    int seed,  // unused in this version (no WHT rotation)
    int bits
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be [N, D]");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");

    const int N = input.size(0);
    const int D = input.size(1);
    const int num_words = (D * bits + 31) / 32;

    // Ensure float32
    auto input_f32 = input.to(torch::kFloat32).contiguous();

    auto radii = torch::empty({N}, torch::dtype(torch::kFloat32).device(input.device()));
    auto packed = torch::zeros({N, num_words}, torch::dtype(torch::kInt32).device(input.device()));

    // Block config: min(D, 256) threads per block, one block per vector
    int threads = std::min(D, 256);
    int num_warps = (threads + 31) / 32;
    int smem = num_warps * sizeof(float);

    auto* inp_ptr = input_f32.data_ptr<float>();
    auto* rad_ptr = radii.data_ptr<float>();
    int32_t* pack_ptr = reinterpret_cast<int32_t*>(packed.data_ptr<int>());

    if (bits == 2) {
        tq_encode_kernel<2><<<N, threads, smem>>>(
            inp_ptr, rad_ptr, pack_ptr, N, D, num_words);
    } else if (bits == 3) {
        tq_encode_kernel<3><<<N, threads, smem>>>(
            inp_ptr, rad_ptr, pack_ptr, N, D, num_words);
    } else {
        tq_encode_kernel<4><<<N, threads, smem>>>(
            inp_ptr, rad_ptr, pack_ptr, N, D, num_words);
    }

    return std::make_tuple(radii, packed);
}


torch::Tensor tq_decode(
    torch::Tensor radii,
    torch::Tensor packed,
    int D,
    int seed,
    int bits
) {
    TORCH_CHECK(radii.is_cuda(), "radii must be on CUDA");
    TORCH_CHECK(packed.is_cuda(), "packed must be on CUDA");
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4, "bits must be 2, 3, or 4");

    const int N = radii.size(0);
    const int num_words = packed.size(1);

    auto output = torch::empty({N, D}, torch::dtype(torch::kFloat32).device(radii.device()));

    int threads = min(D, 256);

    auto* rad_ptr = radii.data_ptr<float>();
    const int32_t* pack_ptr = reinterpret_cast<const int32_t*>(packed.data_ptr<int>());
    auto* out_ptr = output.data_ptr<float>();

    if (bits == 2) {
        tq_decode_kernel<2><<<N, threads>>>(rad_ptr, pack_ptr, out_ptr, N, D, num_words);
    } else if (bits == 3) {
        tq_decode_kernel<3><<<N, threads>>>(rad_ptr, pack_ptr, out_ptr, N, D, num_words);
    } else {
        tq_decode_kernel<4><<<N, threads>>>(rad_ptr, pack_ptr, out_ptr, N, D, num_words);
    }

    return output;
}
