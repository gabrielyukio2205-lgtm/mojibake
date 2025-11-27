#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Configuration
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int D = 64;

// Padding to avoid bank conflicts
// 64 halfs = 128 bytes. Matches shared memory bank width (32 banks * 4 bytes = 128 bytes)
// This causes all rows to start at bank 0.
// We pad by 8 halfs (16 bytes). New stride = 72 halfs.
constexpr int PAD = 8;
constexpr int D_PAD = D + PAD;

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__global__ void flash_attn_wmma_fwd(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ L,
    const float softmax_scale,
    const int N,
    const int stride_b,
    const int stride_h,
    const int stride_n
) {
    // Thread Indices
    int tid = threadIdx.x;
    int laneId = tid % 32;
    int warpId = tid / 32;

    // Grid indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global Pointers
    long long base_offset = (long long)by * (N * D);
    const __half* q_ptr = Q + base_offset;
    const __half* k_ptr = K + base_offset;
    const __half* v_ptr = V + base_offset;
    __half* o_ptr = O + base_offset;

    int q_row_start = bx * 16;

    // Shared Memory Layout
    // We pad Q, K, V, and O scratch.
    // 1. Q: 16x(64+8) half
    // 2. K: 16x(64+8) half
    // 3. V: 16x(64+8) half
    // 4. S: 16x(16+pad) float?
    //    16 floats = 64 bytes.
    //    We can pad S to say 16+4 floats.
    // 5. O_scratch: 16x(64+8) float (larger because float accumulation)

    // Calculate offsets
    // size_half = 16 * 72 = 1152 elements = 2304 bytes.
    int q_offset_sram = 0;
    int k_offset_sram = q_offset_sram + 16 * D_PAD;
    int v_offset_sram = k_offset_sram + 16 * D_PAD;

    // O scratch needs to hold floats.
    // 16 * 72 * 4 bytes = 4608 bytes.
    int o_offset_sram_bytes = (v_offset_sram + 16 * D_PAD) * sizeof(__half);

    extern __shared__ char sram_byte[];
    __half* s_Q = (__half*)sram_byte + q_offset_sram;
    __half* s_K = (__half*)sram_byte + k_offset_sram;
    __half* s_V = (__half*)sram_byte + v_offset_sram;

    // Align O_scratch to 4 bytes
    float* s_O_scratch = (float*)(sram_byte + o_offset_sram_bytes);

    // S aliases first part of O_scratch (it's temporary)
    // Pad S as well. 16 cols + padding.
    // Let's use D_PAD for S as well? 72 floats.
    // Just to keep indexing simple.
    float* s_S = s_O_scratch;

    // Stats
    // Placed after O_scratch
    float* s_m = s_O_scratch + 16 * D_PAD;
    float* s_l = s_m + 16;
    float* s_alpha = s_l + 16;
    __half* s_P = (__half*)(s_alpha + 16); // 16x16 P matrix. Needs padding?
    // s_P is loaded into tensor cores. WMMA load stride needs to be >= 16.
    // Let's pad s_P to 16 rows x 16 cols.
    // Actually WMMA doesn't strictly need padding if stride is handled, but to match logic...
    // Let's keep s_P 16x16 tightly packed for simplicity or pad it 16x(16+8) for bank?
    // 16 halfs = 32 bytes. No bank conflict issue typically for 16x16 load if stride is 16?
    // Let's just use 16 stride for P.

    // Initialize stats
    if (tid < 16) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    // Fragments (Warp 0 only)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];
    if (warpId == 0) {
        #pragma unroll
        for(int i=0; i<4; ++i) {
            wmma::fill_fragment(acc_o[i], 0.0f);
        }
    }

    // --------------------------------------------------------
    // LOAD Q (All Threads)
    // --------------------------------------------------------
    // We want to load 16x64 elements.
    // Destination: s_Q (16 rows, stride 72).
    // Total loads: 1024 elements.
    // 128 threads -> 8 elements per thread (1 float4 load of 2 loaded float4s? No, float4 is 8 bytes = 4 halves).
    // 1024 halves = 256 float4s.
    // 128 threads -> 2 float4 loads per thread.

    int q_global_offset = q_row_start * D;
    const float4* q_global_f4 = (const float4*)(q_ptr + q_global_offset);
    // Writing to padded shared memory is tricky with vectorized stores.
    // s_Q is stride 72. We cannot just cast s_Q to float4* and write linearly.
    // We must write row by row.

    // Parallel Strategy:
    // Each thread handles specific elements.
    // 1024 elements.
    // Linear index mapping:
    // elem_idx = tid + k * 128.
    // row = elem_idx / 64.
    // col = elem_idx % 64.
    // s_Q[row * 72 + col] = loaded_val.
    // But we want vectorized load/store?
    // Vectorized Load from Global (contiguous) is fine.
    // Vectorized Store to Shared (stride 72) is broken if we cross row boundary.
    // But since D=64 (multiple of 8), a float4 (4 halves) fits in a row.
    // So we can do:
    // Load float4 from global.
    // Store float4 to shared at [row * 72 + col].
    // Since D=64, row = idx / 16 (float4s per row).

    // Total 16 rows * 16 float4s = 256 float4s.
    // 128 threads. Each does 2 iters.

    for (int k = 0; k < 2; ++k) {
        int idx_f4 = tid + k * 128; // 0..255
        // Map to row/col
        int row = idx_f4 / 16;
        int col_f4 = idx_f4 % 16;

        if (row < 16) {
            // Global Load
            // Bounds check
            if (q_row_start * D + idx_f4 * 4 < N * D) { // *4 because float4 is 4 halves? Wait.
                 // float4 = 16 bytes. half = 2 bytes.
                 // float4 = 8 halves.
                 // So D=64 halves = 8 float4s.
                 // So row = idx_f4 / 8.
                 // Total float4s = 16 rows * 8 = 128.
                 // 128 threads. Exactly 1 load per thread!
            }
        }
    }

    // Correct logic for 1 load per thread (since 128 threads, 16x64 tile = 128 float4s)
    // float4 = 8 halves.
    int idx_f4 = tid; // 0..127
    int row = idx_f4 / 8; // 0..15
    int col_f4 = idx_f4 % 8; // 0..7
    int col_half = col_f4 * 8; // 0..56

    if (idx_f4 < 128) {
        // Global address
        if (q_row_start * D + idx_f4 * 8 < N * D) { // Check safe
            float4 val = q_global_f4[idx_f4]; // Load contiguous

            // Shared store (Padded)
            // Cast padded ptr
            // s_Q is half*. Address: s_Q + row*72 + col_half
            // We cast that address to float4*.
            *(float4*)(&s_Q[row * D_PAD + col_half]) = val;
        } else {
             // Zero fill shared mem
             *(float4*)(&s_Q[row * D_PAD + col_half]) = make_float4(0,0,0,0);
        }
    }

    __syncthreads();

    // Loop over K chunks
    for (int k_idx = 0; k_idx < (N + 15) / 16; ++k_idx) {
        int k_row_start = k_idx * 16;

        // Load K, V (All Threads)
        // Same logic: 1 float4 per thread
        int global_k_idx_f4 = (k_row_start * D) / 8; // Offset in float4s

        if (tid < 128) {
             int row = tid / 8;
             int col_half = (tid % 8) * 8;

             // K
             if (k_row_start * D + tid * 8 < N * D) {
                 float4 val = ((const float4*)k_ptr)[global_k_idx_f4 + tid];
                 *(float4*)(&s_K[row * D_PAD + col_half]) = val;
             } else {
                 *(float4*)(&s_K[row * D_PAD + col_half]) = make_float4(0,0,0,0);
             }

             // V
             if (k_row_start * D + tid * 8 < N * D) {
                 float4 val = ((const float4*)v_ptr)[global_k_idx_f4 + tid];
                 *(float4*)(&s_V[row * D_PAD + col_half]) = val;
             } else {
                 *(float4*)(&s_V[row * D_PAD + col_half]) = make_float4(0,0,0,0);
             }
        }
        __syncthreads();

        // --------------------------------------------------------
        // COMPUTE (Warp 0 Only)
        // --------------------------------------------------------
        if (warpId == 0) {
            // 1. Compute S = Q * K^T
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
            wmma::fill_fragment(acc_s, 0.0f);

            for (int d_chunk = 0; d_chunk < 4; ++d_chunk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
                // Load from padded s_Q. Stride 72.
                wmma::load_matrix_sync(a_frag, s_Q + d_chunk * 16, D_PAD);

                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
                // Load from padded s_K. Stride 72. Transpose via col_major.
                wmma::load_matrix_sync(b_frag, s_K + d_chunk * 16, D_PAD);

                wmma::mma_sync(acc_s, a_frag, b_frag, acc_s);
            }

            // 2. Softmax
            // Store S to Shared (float). Reuse s_S (padded).
            wmma::store_matrix_sync(s_S, acc_s, D_PAD, wmma::mem_row_major);
        }

        // Wait for S store
        __syncthreads();

        // 2b. CPU Softmax Math (Warp 0)
        // Could be done by all threads if carefully mapped, but Warp 0 is enough and safer.
        if (warpId == 0) {
            if (laneId < 16) {
                int row = laneId;
                float row_max = -INFINITY;
                // Find Max
                for (int c = 0; c < 16; ++c) {
                    float val = s_S[row * D_PAD + c];
                    val *= softmax_scale;
                    s_S[row * D_PAD + c] = val;
                    if (val > row_max) row_max = val;
                }

                float old_m = s_m[row];
                float new_m = max(old_m, row_max);
                float alpha = fast_exp(old_m - new_m);

                s_alpha[row] = alpha;
                s_m[row] = new_m;

                float row_sum = 0.0f;
                for (int c = 0; c < 16; ++c) {
                    float val = s_S[row * D_PAD + c];
                    float e = fast_exp(val - new_m);
                    s_S[row * D_PAD + c] = e;
                    // Also convert to half for P
                    s_P[row * 16 + c] = __float2half(e); // s_P is packed 16x16? Yes.
                    row_sum += e;
                }

                s_l[row] = s_l[row] * alpha + row_sum;
            }
        }
        __syncthreads(); // Wait for s_alpha and s_P

        if (warpId == 0) {
            // 2c. Rescale Acc_O
            // Store acc_o to s_O_scratch (padded)
            for(int i=0; i<4; ++i) {
                wmma::store_matrix_sync(s_O_scratch + i*16, acc_o[i], D_PAD, wmma::mem_row_major);
            }
        }
        __syncthreads(); // Wait for store

        // Parallel Scaling (All threads in Warp 0? Or All threads?)
        // Warp 0 is already active. Let's just use Warp 0 to avoid sync overhead complexity?
        // But 32 threads scaling 1024 elements = 32 iters.
        // If we use 128 threads -> 8 iters. Faster.
        // Let's use all threads for this ALUM op.

        // idx goes 0..1024 (16*64).
        // Stride is D_PAD (72).
        // Logic: row = idx/64... No, memory is padded.
        // We need to iterate logical 16x64, but map to physical stride.

        int total_elems = 16 * 64;
        for (int i = tid; i < total_elems; i += 128) {
            int r = i / 64;
            int c = i % 64;
            float alpha = s_alpha[r];
            s_O_scratch[r * D_PAD + c] *= alpha;
        }
        __syncthreads();

        if (warpId == 0) {
            // Load back acc_o
            for(int i=0; i<4; ++i) {
                wmma::load_matrix_sync(acc_o[i], s_O_scratch + i*16, D_PAD, wmma::mem_row_major);
            }

            // 3. Compute O += P * V
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
            // s_P is packed 16x16
            wmma::load_matrix_sync(p_frag, s_P, 16);

            for (int i = 0; i < 4; ++i) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
                // s_V is padded
                wmma::load_matrix_sync(v_frag, s_V + i*16, D_PAD);
                wmma::mma_sync(acc_o[i], p_frag, v_frag, acc_o[i]);
            }
        }
        __syncthreads();
    } // End K loop

    // Write Output (Warp 0 store to Shared, then All Threads Global write)
    if (warpId == 0) {
        for(int i=0; i<4; ++i) {
            wmma::store_matrix_sync(s_O_scratch + i*16, acc_o[i], D_PAD, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Global Write (All Threads)
    // 16x64 elements.
    for (int i = tid; i < 16 * 64; i += 128) {
        int r = i / 64;
        int c = i % 64;

        if (q_row_start + r < N) { // Check bounds
             float val = s_O_scratch[r * D_PAD + c];
             float l = s_l[r];
             val /= (l + 1e-6f);

             // Global address
             int global_idx = (q_row_start + r) * D + c;
             // Bound check D? (Assumed D=64)
             o_ptr[global_idx] = __float2half(val);
        }
    }
}
