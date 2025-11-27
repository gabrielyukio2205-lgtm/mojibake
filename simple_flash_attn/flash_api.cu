#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int D = 64;

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__global__ void flash_attn_fwd_kernel(
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
    int bx = blockIdx.x;
    int by = blockIdx.y;

    long long offset = (long long)by * (N * D);
    const __half* q_ptr = Q + offset;
    const __half* k_ptr = K + offset;
    const __half* v_ptr = V + offset;
    __half* o_ptr = O + offset;
    float* l_ptr = L + (long long)by * N;

    extern __shared__ __half sram[];
    __half* s_Q = sram;
    __half* s_K = s_Q + Br * D;
    __half* s_V = s_K + Bc * D;

    int tx = threadIdx.x;
    int my_row = tx / 2;
    int my_split = tx % 2; // 0 or 1
    int col_start = my_split * 32;

    float acc_o[32];
    for (int i = 0; i < 32; ++i) acc_o[i] = 0.0f;

    float acc_m = -INFINITY;
    float acc_l = 0.0f;

    int q_start_row = bx * Br;

    // Load Q
    // 128 threads * 8 (float4 x 2) = 1024 elems per iter. 4096 / 1024 = 4.
    const float4* q_ptr_f4 = (const float4*)q_ptr;
    float4* s_Q_f4 = (float4*)s_Q;
    int q_block_offset_f4 = (q_start_row * D) / 8;

    for (int i = 0; i < 4; ++i) {
        int idx = i * 128 + tx;
        if ((q_start_row * D) / 8 + idx < (N * D) / 8) {
             s_Q_f4[idx] = q_ptr_f4[q_block_offset_f4 + idx];
        } else {
             s_Q_f4[idx] = make_float4(0,0,0,0);
        }
    }
    __syncthreads();

    // Loop K blocks
    for (int k_idx = 0; k_idx < (N + Bc - 1) / Bc; ++k_idx) {
        int k_start_row = k_idx * Bc;
        int k_offset_f4 = (k_start_row * D) / 8;

        const float4* k_ptr_f4 = (const float4*)k_ptr;
        const float4* v_ptr_f4 = (const float4*)v_ptr;
        float4* s_K_f4 = (float4*)s_K;
        float4* s_V_f4 = (float4*)s_V;

        for (int i = 0; i < 4; ++i) {
            int idx = i * 128 + tx;
             // Bound check
             if ((k_start_row * D)/8 + idx < (N * D)/8) {
                s_K_f4[idx] = k_ptr_f4[k_offset_f4 + idx];
                s_V_f4[idx] = v_ptr_f4[k_offset_f4 + idx];
             } else {
                s_K_f4[idx] = make_float4(0,0,0,0);
                s_V_f4[idx] = make_float4(0,0,0,0);
             }
        }
        __syncthreads();

        // Compute
        if (my_row < Br && (q_start_row + my_row) < N) {
            for (int j = 0; j < Bc; ++j) {
                if (k_start_row + j >= N) break;

                float score = 0.0f;
                // Dot product
                // Accessing 2D shared array with flat pointer arithmetic
                // s_Q[my_row][col_start + k] -> s_Q[my_row * D + col_start + k]
                #pragma unroll
                for (int k = 0; k < 32; ++k) {
                     score += __half2float(s_Q[my_row * D + col_start + k]) * __half2float(s_K[j * D + col_start + k]);
                }

                // Reduce split
                score += __shfl_xor_sync(0xffffffff, score, 1);
                score *= softmax_scale;

                float old_m = acc_m;
                float new_m = max(old_m, score);
                float alpha = fast_exp(old_m - new_m);
                float beta = fast_exp(score - new_m);

                acc_m = new_m;
                acc_l = acc_l * alpha + beta;

                #pragma unroll
                for (int k = 0; k < 32; ++k) {
                    acc_o[k] = acc_o[k] * alpha + beta * __half2float(s_V[j * D + col_start + k]);
                }
            }
        }
        __syncthreads();
    }

    // Write Output
    if (my_row < Br && (q_start_row + my_row) < N) {
        // Write L
        if (my_split == 0) { // Avoid race
             // L is usually logsumexp: L = m + log(l)
            l_ptr[q_start_row + my_row] = acc_m + logf(acc_l);
        }

        // Write O
        // Each thread has 32 floats (half row).
        // Convert to half and store.
        int row_offset = (q_start_row + my_row) * D;
        for (int k = 0; k < 32; ++k) {
            o_ptr[row_offset + col_start + k] = __float2half(acc_o[k] / acc_l);
        }
    }
}

torch::Tensor flash_attn_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    // D is hardcoded 64 in kernel for simplicity

    auto O = torch::zeros_like(Q);
    auto L = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    int grid_n = (N + Br - 1) / Br;
    dim3 grid(grid_n, B * H);
    dim3 block(128);

    int shared_mem_size = (Br*D + Bc*D + Bc*D) * sizeof(__half); // 64*64*3 * 2 = 24KB

    flash_attn_fwd_kernel<<<grid, block, shared_mem_size>>>(
        (const __half*)Q.data_ptr(),
        (const __half*)K.data_ptr(),
        (const __half*)V.data_ptr(),
        (__half*)O.data_ptr(),
        (float*)L.data_ptr(),
        softmax_scale,
        N,
        0, 0, 0 // Strides not used in simplified kernel
    );

    return O;
}
