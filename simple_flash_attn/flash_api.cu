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
constexpr int PAD = 8;
constexpr int D_PAD = D + PAD; // 72

// Block sizes
constexpr int Br = 64; // Process 64 rows of Q per block (4 Warps)
constexpr int Bc = 16; // Process 16 rows of K/V per step

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
    int warpId = tid / 32; // 0..3

    // Grid indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global Pointers
    long long base_offset = (long long)by * (N * D);
    const __half* q_ptr = Q + base_offset;
    const __half* k_ptr = K + base_offset;
    const __half* v_ptr = V + base_offset;
    __half* o_ptr = O + base_offset;
    float* l_ptr = L + (long long)by * N;

    int q_row_start = bx * Br; // bx * 64

    // Shared Memory Layout
    // 1. Q: Br x D_PAD = 64 x 72 half = 4608 * 2 = 9216 bytes (~9KB)
    // 2. K: Bc x D_PAD = 16 x 72 half = 2304 bytes (~2.2KB)
    // 3. V: Bc x D_PAD = 16 x 72 half = 2304 bytes (~2.2KB)
    // 4. O_scratch: Br x D_PAD float = 64 x 72 * 4 = 18432 bytes (~18KB)

    extern __shared__ char sram_byte[];

    __half* s_Q = (__half*)sram_byte; // 9KB
    __half* s_K = s_Q + Br * D_PAD;   // 2KB
    __half* s_V = s_K + Bc * D_PAD;   // 2KB

    float* s_O_scratch = (float*)(s_V + Bc * D_PAD); // 18KB

    // Scratch for Softmax Stats (m, l, alpha)
    // We need per-row stats. Br=64 rows.
    float* s_m = s_O_scratch + Br * D_PAD; // 64 floats
    float* s_l = s_m + Br;                 // 64 floats
    float* s_alpha = s_l + Br;             // 64 floats

    // Scratch for S (Score) and P (Prob)
    // Each warp processes 16x16. We can reuse memory or allocate per-warp.
    // If we want to store S to shared, we need 16x16 * 4 warps? No, S is transient.
    // But Softmax requires storing S.
    // Let's allocate S_scratch: Br x (16+PAD) float?
    // Wait, K loop step is 16. So we compute 64x16 matrix S.
    // Yes. S is 64x16.
    // Let's allocate S: 64 x 16 floats.
    // Reuse s_O_scratch?
    // s_O_scratch is 18KB. We need it to persist O across K loops.
    // So we cannot overlap S with O_scratch.
    // Allocate S after stats.
    float* s_S = s_alpha + Br; // 64 * 16 float = 1024 * 4 = 4KB.
    __half* s_P = (__half*)(s_S + Br * 16); // 64 * 16 half = 2KB.

    // Total: 9+2+2 + 18 + small + 4 + 2 = ~37KB. Fits in 48KB.

    // Initialize stats
    // 128 threads. 64 rows.
    if (tid < 64) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    // Fragments
    // Each warp maintains accumulators for its 16 rows.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];
    #pragma unroll
    for(int i=0; i<4; ++i) {
        wmma::fill_fragment(acc_o[i], 0.0f);
    }

    // --------------------------------------------------------
    // LOAD Q (All Threads)
    // --------------------------------------------------------
    // Br x D = 64 x 64 = 4096 elements.
    // 128 threads. 32 elements/thread = 4 float4 loads/thread.
    // Destination stride: D_PAD (72).

    int q_base_global = q_row_start * D;
    const float4* q_global_ptr = (const float4*)(q_ptr + q_base_global);

    for (int k = 0; k < 4; ++k) {
        int idx = tid + k * 128; // 0..511
        // Each idx handles 1 float4 (8 halves).
        // Row = idx / 8.
        // Col_half = (idx % 8) * 8.
        int row = idx / 8;
        int col_half = (idx % 8) * 8;

        if (row < Br) {
            if (q_row_start * D + idx * 8 < N * D) {
                float4 val = q_global_ptr[idx];
                *(float4*)(&s_Q[row * D_PAD + col_half]) = val;
            } else {
                *(float4*)(&s_Q[row * D_PAD + col_half]) = make_float4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    // Loop over K chunks (step 16)
    for (int k_idx = 0; k_idx < (N + 15) / 16; ++k_idx) {
        int k_row_start = k_idx * 16;

        // Load K, V (16 rows each)
        // 16 x 64 = 1024 elements each.
        // 128 threads. 8 elements/thread = 1 float4/thread.

        const float4* k_g = (const float4*)(k_ptr + k_row_start * D);
        const float4* v_g = (const float4*)(v_ptr + k_row_start * D);

        // Use all threads
        int idx = tid; // 0..127
        int row = idx / 8; // 0..15
        int col_half = (idx % 8) * 8;

        // Load K
        if (idx < 128) { // 16 rows * 8 float4s
             if (k_row_start * D + idx*8 < N * D) {
                  float4 val = k_g[idx];
                  *(float4*)(&s_K[row * D_PAD + col_half]) = val;
             } else {
                  *(float4*)(&s_K[row * D_PAD + col_half]) = make_float4(0,0,0,0);
             }
        }

        // Load V
        if (idx < 128) {
             if (k_row_start * D + idx*8 < N * D) {
                  float4 val = v_g[idx];
                  *(float4*)(&s_V[row * D_PAD + col_half]) = val;
             } else {
                  *(float4*)(&s_V[row * D_PAD + col_half]) = make_float4(0,0,0,0);
             }
        }
        __syncthreads();

        // --------------------------------------------------------
        // COMPUTE (All Warps Parallel)
        // --------------------------------------------------------
        // Warp w handles Q rows [w*16, (w+1)*16).

        int my_row_offset = warpId * 16; // 0, 16, 32, 48

        // 1. Compute S_w = Q_w * K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);

        for (int d_chunk = 0; d_chunk < 4; ++d_chunk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            // Load Q: start at my_row_offset. Stride D_PAD.
            wmma::load_matrix_sync(a_frag, s_Q + my_row_offset * D_PAD + d_chunk * 16, D_PAD);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            // Load K: start at 0. Stride D_PAD.
            wmma::load_matrix_sync(b_frag, s_K + d_chunk * 16, D_PAD);

            wmma::mma_sync(acc_s, a_frag, b_frag, acc_s);
        }

        // 2. Softmax
        // Store S_w to Shared
        // S_w is 16x16.
        // Destination s_S is 64x16.
        // Offset: my_row_offset * 16.
        // Stride: 16. (Packed)
        wmma::store_matrix_sync(s_S + my_row_offset * 16, acc_s, 16, wmma::mem_row_major);

        // We can sync warp only? No, we need to wait for store if we read it?
        // We read what we wrote. Syncwarp is enough.
        // But s_S is in shared.
        // Since warps work on disjoint s_S regions, we don't strictly need global sync, but __syncwarp is good.
        // Actually, we need syncthreads for the next step? No.

        // Compute Softmax stats
        // Each thread in warp handles its row (0..15 relative to warp).
        if (laneId < 16) {
            int row_rel = laneId;
            int row_abs = my_row_offset + row_rel;

            float row_max = -INFINITY;
            for (int c = 0; c < 16; ++c) {
                float val = s_S[row_abs * 16 + c];
                val *= softmax_scale;
                s_S[row_abs * 16 + c] = val;
                if (val > row_max) row_max = val;
            }

            float old_m = s_m[row_abs];
            float new_m = max(old_m, row_max);
            float alpha = fast_exp(old_m - new_m);

            s_alpha[row_abs] = alpha;
            s_m[row_abs] = new_m;

            float row_sum = 0.0f;
            for (int c = 0; c < 16; ++c) {
                float val = s_S[row_abs * 16 + c];
                float e = fast_exp(val - new_m);
                s_S[row_abs * 16 + c] = e;
                // Pack to half for P
                s_P[row_abs * 16 + c] = __float2half(e);
                row_sum += e;
            }

            s_l[row_abs] = s_l[row_abs] * alpha + row_sum;
        }
        // Need memory visibility for s_alpha?
        // Threads in same warp need to see s_alpha written by laneId.
        // __syncwarp() ensures register visibility? No, shared mem visibility.
        // Within a warp, shared mem is coherent.

        // 2c. Rescale Acc_O
        // Store acc_o -> s_O_scratch
        // Each warp stores its 16 rows.
        for(int i=0; i<4; ++i) {
             // s_O_scratch: 64x72.
             // offset: my_row_offset * 72 + i*16 (cols).
             // stride: 72.
             wmma::store_matrix_sync(s_O_scratch + my_row_offset * D_PAD + i*16, acc_o[i], D_PAD, wmma::mem_row_major);
        }
        // Barrier needed?
        // We are reading back what we wrote.
        // And we do scaling.
        // Warp handles its own data.

        // Scale O
        // 32 threads in warp.
        // Rows: my_row_offset .. my_row_offset+15.
        // Cols: 0..63.
        // Total 16 * 64 = 1024 elements per warp.
        // 32 threads -> 32 elems/thread.
        for (int k = 0; k < 32; ++k) {
             int idx = k * 32 + laneId; // 0..1023
             int r_rel = idx / 64;
             int c = idx % 64;

             int r_abs = my_row_offset + r_rel;
             float alpha = s_alpha[r_abs];

             s_O_scratch[r_abs * D_PAD + c] *= alpha;
        }

        // Load back acc_o
        for(int i=0; i<4; ++i) {
             wmma::load_matrix_sync(acc_o[i], s_O_scratch + my_row_offset * D_PAD + i*16, D_PAD, wmma::mem_row_major);
        }

        // 3. Compute O += P * V
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
        // P is 16x16 (packed stride 16).
        wmma::load_matrix_sync(p_frag, s_P + my_row_offset * 16, 16);

        for (int i = 0; i < 4; ++i) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
            // V is broadcast (shared across warps). Offset 0.
            // s_V padded stride.
            wmma::load_matrix_sync(v_frag, s_V + i*16, D_PAD);
            wmma::mma_sync(acc_o[i], p_frag, v_frag, acc_o[i]);
        }

        // Sync threads for next K iteration
        // Must ensure s_K / s_V are essentially "free" to be overwritten.
        // We finished using them.
        __syncthreads();
    }

    // Write Output
    // Store O to Shared
    for(int i=0; i<4; ++i) {
         wmma::store_matrix_sync(s_O_scratch + (warpId * 16) * D_PAD + i*16, acc_o[i], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    // Global Write (All Threads, Coalesced)
    // 64 x 64 elements = 4096.
    // 128 threads. 32 per thread.
    // Vectorized logic?
    // s_O_scratch is float. o_ptr is half.
    // Manual cast.

    for (int k = 0; k < 32; ++k) {
         int idx = tid + k * 128;
         if (idx < Br * D) {
             int r = idx / D; // Logical row (0..63)
             int c = idx % D;

             // Remap to padded shared
             float val = s_O_scratch[r * D_PAD + c];

             // Normalize
             int global_row = q_row_start + r;
             if (global_row < N) {
                 float l = s_l[r];
                 o_ptr[global_row * D + c] = __float2half(val / (l + 1e-6f));

                 // Write L (LogSumExp) once per row
                 if (c == 0) {
                     l_ptr[global_row] = s_m[r] + logf(l + 1e-6f);
                 }
             }
         }
    }
}
