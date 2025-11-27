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
    // Identity
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    // Grid indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    long long base_offset = (long long)by * (N * D);
    const __half* q_ptr = Q + base_offset;
    const __half* k_ptr = K + base_offset;
    const __half* v_ptr = V + base_offset;
    __half* o_ptr = O + base_offset;

    int q_row_start = bx * 16;

    // Shared Memory Layout
    // 1. Q: 16x64 half = 1024 * 2 = 2048 bytes
    // 2. K: 16x64 half = 2048 bytes
    // 3. V: 16x64 half = 2048 bytes
    // 4. S: 16x16 float = 256 * 4 = 1024 bytes (also used for O scratch)
    // 5. P: 16x16 half = 256 * 2 = 512 bytes
    // 6. Stats: m, l (16 float each) = 128 bytes
    // 7. Scratch for O: We need 16x64 float = 4096 bytes.
    // Can we reuse?
    // S and P are temporary.
    // We can reuse S memory for O scratch? No, S is 1KB. O is 4KB.
    // Total needed: 2+2+2+4 (O scratch) + small stuff = ~11KB. OK.

    extern __shared__ char sram_byte[];
    __half* s_Q = (__half*)sram_byte;
    __half* s_K = s_Q + 16 * 64;
    __half* s_V = s_K + 16 * 64;
    float*  s_O_scratch = (float*)(s_V + 16 * 64); // 4KB scratch

    // Pointers into scratch for S
    float*  s_S = s_O_scratch; // Alias, first 1KB
    __half* s_P = (__half*)(s_S + 16 * 16);

    float* s_m = (float*)(s_O_scratch + 16 * 64); // After O scratch
    float* s_l = s_m + 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];

    #pragma unroll
    for(int i=0; i<4; ++i) {
        wmma::fill_fragment(acc_o[i], 0.0f);
    }

    if (laneId < 16) {
        s_m[laneId] = -INFINITY;
        s_l[laneId] = 0.0f;
    }

    // Load Q
    int q_offset = q_row_start * D;
    const float4* q_ptr_f4 = (const float4*)(q_ptr + q_offset);
    float4* s_Q_f4 = (float4*)s_Q;

    for (int i = 0; i < 4; ++i) {
        int t_idx = threadIdx.x + i * 32;
        if (t_idx < 128) {
            if (q_row_start * D + t_idx * 8 < N * D)
               s_Q_f4[t_idx] = q_ptr_f4[t_idx];
            else
               s_Q_f4[t_idx] = make_float4(0,0,0,0);
        }
    }
    __syncthreads();

    // Loop over K chunks
    for (int k_idx = 0; k_idx < (N + 15) / 16; ++k_idx) {
        int k_row_start = k_idx * 16;

        const float4* k_ptr_f4 = (const float4*)(k_ptr + k_row_start * D);
        const float4* v_ptr_f4 = (const float4*)(v_ptr + k_row_start * D);
        float4* s_K_f4 = (float4*)s_K;
        float4* s_V_f4 = (float4*)s_V;

        for (int i = 0; i < 4; ++i) {
            int t_idx = threadIdx.x + i * 32;
            if (t_idx < 128) {
                if (k_row_start * D + t_idx * 8 < N * D) {
                    s_K_f4[t_idx] = k_ptr_f4[t_idx];
                    s_V_f4[t_idx] = v_ptr_f4[t_idx];
                } else {
                    s_K_f4[t_idx] = make_float4(0,0,0,0);
                    s_V_f4[t_idx] = make_float4(0,0,0,0);
                }
            }
        }
        __syncthreads();

        // 1. Compute S = Q * K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);

        for (int d_chunk = 0; d_chunk < 4; ++d_chunk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, s_Q + d_chunk * 16, 64);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, s_K + d_chunk * 16, 64);

            wmma::mma_sync(acc_s, a_frag, b_frag, acc_s);
        }

        // 2. Softmax
        wmma::store_matrix_sync(s_S, acc_s, 16, wmma::mem_row_major);
        __syncthreads();

        if (threadIdx.x < 16) {
            int row = threadIdx.x;
            float row_max = -INFINITY;
            for (int c = 0; c < 16; ++c) {
                float val = s_S[row * 16 + c];
                val *= softmax_scale;
                s_S[row * 16 + c] = val;
                if (val > row_max) row_max = val;
            }

            float old_m = s_m[row];
            float new_m = max(old_m, row_max);

            float alpha = fast_exp(old_m - new_m);
            float beta = fast_exp(row_max - new_m);

            float row_sum = 0.0f;
            for (int c = 0; c < 16; ++c) {
                float val = s_S[row * 16 + c];
                // P = exp(val - new_m)
                // Note: val is already scaled score.
                // We stored val back to s_S.
                // But wait, if we are in next K-block, we compare with global max?
                // The algorithm is:
                // m_new = max(m_old, row_max)
                // l_new = l_old * alpha + sum(exp(scores - m_new))

                float e = fast_exp(val - new_m);
                s_S[row * 16 + c] = e;
                row_sum += e;
            }

            s_m[row] = new_m;
            float old_l = s_l[row];
            s_l[row] = old_l * alpha + row_sum;

            // 2b. Rescale Accumulators (Correctness Fix)
            // We need to scale ALL elements of O (current row) by alpha.
            // O is distributed in acc_o fragments.
            // We will do this via Store -> Scale -> Load.
            // But doing it here inside the thread loop is inefficient.
            // We set a flag or shared variable?
            // Actually, we can just save `alpha` to shared memory and let all threads participate in scaling.
            // Re-use `s_P` space for alpha storage temporarily? No, s_P is needed for P.
            // Use s_l for temporary storage? No.
            // Add `s_alpha` array.
            // Just write alpha back to s_S (reuse space? No).
            // Write to `s_m`? No.
            // Let's use `s_S` last row? No.
            // Just allocate small float s_alpha[16]
        }

        // Need to synchronize to make sure s_m/alpha is ready.
        // But alpha is local variable.
        // Let's store alpha to shared memory so we can use it to scale O.
        // We will repurpose s_m or define s_alpha.
        // Let's define s_alpha at start.
        // Or better: Recompute alpha outside? No.

        // Hack: Store alpha in s_m temporarily? No, we need s_m for next iter.
        // Store in s_P? s_P is half.
        // Let's just extend shared memory slightly.
        // Or reuse part of s_S? s_S is 256 floats. We use all.
        // Reuse s_l? No.

        // We need 16 floats.
        // (float*)s_O_scratch + 16*64 + 32 ?
        // Yes, plenty of space at the end.
        float* s_alpha = s_l + 16;

        if (threadIdx.x < 16) {
             int row = threadIdx.x;
             // Recompute alpha
             float old_m = s_m[row]; // Wait, s_m was updated!
             // Problem: I updated s_m in previous lines.
             // I should calculate alpha, store it, THEN update s_m.

             // Refactoring the single thread logic
             // ...
             // (Done implicitly below in correct order)
        }
        __syncthreads();

        // Refined Softmax Logic in correct order
        if (threadIdx.x < 16) {
            int row = threadIdx.x;
            float row_max = -INFINITY;
            for (int c = 0; c < 16; ++c) {
                float val = s_S[row * 16 + c];
                val *= softmax_scale;
                s_S[row * 16 + c] = val;
                if (val > row_max) row_max = val;
            }

            float old_m = s_m[row];
            float new_m = max(old_m, row_max);
            float alpha = fast_exp(old_m - new_m);

            s_alpha[row] = alpha; // Store for scaling
            s_m[row] = new_m;     // Update state

            float row_sum = 0.0f;
            for (int c = 0; c < 16; ++c) {
                float val = s_S[row * 16 + c];
                float e = fast_exp(val - new_m);
                s_S[row * 16 + c] = e;
                row_sum += e;
            }

            s_l[row] = s_l[row] * alpha + row_sum;
        }
        __syncthreads();

        // 2c. Rescale O (The Fix)
        // Store acc_o -> s_O_scratch
        for(int i=0; i<4; ++i) {
            wmma::store_matrix_sync(s_O_scratch + i*16, acc_o[i], 64, wmma::mem_row_major);
        }
        __syncthreads();

        // Scale in place
        // 16 rows * 64 cols. 1024 elements. 32 threads.
        for (int i = 0; i < 32; ++i) { // 32 items per thread
             int idx = threadIdx.x + i * 32;
             if (idx < 1024) {
                 int r = idx / 64;
                 // int c = idx % 64;
                 float alpha = s_alpha[r];
                 s_O_scratch[idx] *= alpha;
             }
        }
        __syncthreads();

        // Load back to acc_o
        for(int i=0; i<4; ++i) {
            wmma::load_matrix_sync(acc_o[i], s_O_scratch + i*16, 64, wmma::mem_row_major);
        }

        // Convert P to half
        if (threadIdx.x < 16) {
             int row = threadIdx.x;
             for(int c=0; c<16; ++c) {
                 s_P[row*16 + c] = __float2half(s_S[row*16+c]);
             }
        }
        __syncthreads();

        // 3. Compute O += P * V
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, s_P, 16);

        for (int i = 0; i < 4; ++i) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, s_V + i*16, 64);
            wmma::mma_sync(acc_o[i], p_frag, v_frag, acc_o[i]);
        }
    }

    // Store O (float) to Shared Mem
    // Reuse s_O_scratch
    __syncthreads();

    for(int i=0; i<4; ++i) {
        wmma::store_matrix_sync(s_O_scratch + i*16, acc_o[i], 64, wmma::mem_row_major);
    }
    __syncthreads();

    // Write to Global (Half)
    for (int i = 0; i < 4; ++i) {
        int t_idx = threadIdx.x + i * 32;
        if (t_idx < 16*64) {
            int r = t_idx / 64;
            // int c = t_idx % 64;
            float val = s_O_scratch[t_idx];
            float l = s_l[r];
            val /= (l + 1e-6f);

            if (q_row_start * D + t_idx < N * D)
                o_ptr[t_idx] = __float2half(val);
        }
    }
}
