#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

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

constexpr int Br = 64;
constexpr int Bc = 16;

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__global__ void flash_attn_wmma_pipeline(
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
    // Pipeline object
    // Requires sm_70+ (Volta) but cuda_pipeline is better on Ampere (sm_80).
    // For sm_75 (Turing) or sm_70, memcpy_async might fallback or work differently, but API exists.
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Indices
    int tid = threadIdx.x;
    int laneId = tid % 32;
    int warpId = tid / 32; // 0..3

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global Pointers
    long long base_offset = (long long)by * (N * D);
    const __half* q_ptr = Q + base_offset;
    const __half* k_ptr = K + base_offset;
    const __half* v_ptr = V + base_offset;
    __half* o_ptr = O + base_offset;
    float* l_ptr = L + (long long)by * N;

    int q_row_start = bx * Br;

    // Shared Memory Layout
    // Double Buffer for K and V.
    // 1. Q: Br x D_PAD (No double buffer needed, fixed per block)
    // 2. K: 2 * Bc x D_PAD
    // 3. V: 2 * Bc x D_PAD
    // 4. O_scratch, Stats...

    extern __shared__ char sram_byte[];

    __half* s_Q = (__half*)sram_byte;
    // Q size: 64 * 72 * 2 = 9216 bytes.

    int q_size_half = Br * D_PAD;

    // K double buffer
    // s_K[0] at start + Q size
    __half* s_K_base = s_Q + q_size_half;
    // V double buffer
    // s_V_base after K buffers
    // K buffer size: 2 * 16 * 72 = 2304 elements.
    int kv_buffer_size = 2 * Bc * D_PAD;
    __half* s_V_base = s_K_base + kv_buffer_size;

    float* s_O_scratch = (float*)(s_V_base + kv_buffer_size);

    float* s_m = s_O_scratch + Br * D_PAD;
    float* s_l = s_m + Br;
    float* s_alpha = s_l + Br;
    float* s_S = s_alpha + Br;
    __half* s_P = (__half*)(s_S + Br * 16);

    // Pointers for Double Buffering
    // K0 = s_K_base, K1 = s_K_base + Bc*D_PAD
    int stage_stride = Bc * D_PAD;

    // Initialize stats
    if (tid < 64) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    // Fragments
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];
    #pragma unroll
    for(int i=0; i<4; ++i) {
        wmma::fill_fragment(acc_o[i], 0.0f);
    }

    // --------------------------------------------------------
    // LOAD Q (Async? No, Standard Load once)
    // --------------------------------------------------------
    // We could use async copy here too but Q is loaded once.
    // Let's use standard vectorized load we had.

    int q_base_global = q_row_start * D;
    const float4* q_global_ptr = (const float4*)(q_ptr + q_base_global);

    for (int k = 0; k < 4; ++k) {
        int idx = tid + k * 128;
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
    __syncthreads(); // Q ready

    // Pipeline Logic
    int total_k_blocks = (N + 15) / 16;

    int write_stage = 0;
    int read_stage = 0;

    // --------------------------------------------------------
    // PROLOGUE: Load Block 0
    // --------------------------------------------------------
    if (total_k_blocks > 0) {
        int k_idx = 0;
        int k_row_start = 0;

        // Async Load K0, V0
        pipe.producer_acquire();
        // Load to s_K[write_stage], s_V[write_stage]
        // Use all threads to issue copy commands

        // Calc pointers
        __half* dst_k = s_K_base + write_stage * stage_stride;
        __half* dst_v = s_V_base + write_stage * stage_stride;
        const __half* src_k = k_ptr + k_row_start * D;
        const __half* src_v = v_ptr + k_row_start * D;

        // 16 rows * 64 cols = 1024 elements = 2048 bytes.
        // We use memcpy_async.
        // Note: memcpy_async copies bytes.
        // Each thread copies a chunk?
        // cuda::memcpy_async can be called by each thread for a piece.
        // 128 threads. Each copies 16 bytes (float4).

        // Logic: Same idx mapping as before
        int idx = tid; // 0..127
        int row = idx / 8;
        int col_half = (idx % 8) * 8;
        int src_offset_half = idx * 8;

        // Check bounds
        if (k_row_start * D + src_offset_half < N * D) {
            // Address of 16 bytes
            // Note: aligned access required. D=64 is aligned.
            cuda::memcpy_async(
                &dst_k[row * D_PAD + col_half],
                &src_k[src_offset_half],
                16, // 16 bytes = 8 halves = 1 float4
                pipe
            );
             cuda::memcpy_async(
                &dst_v[row * D_PAD + col_half],
                &src_v[src_offset_half],
                16,
                pipe
            );
        } else {
             // Zero fill? memcpy_async doesn't zero fill from OOB.
             // We must zero fill manually or mask.
             // Since async, we can't easily "write zero" unless we do memset.
             // But wait, the previous logic did explicit write.
             // For simplicity, we assume padding or safe bounds.
             // If unsafe, we should write zero.
             // Writing zero via register store is fine.
             // But we are in "producer_acquire".
             // We can mix async copy and sync store? Yes, but need to be careful.
             *(float4*)(&dst_k[row * D_PAD + col_half]) = make_float4(0,0,0,0);
             *(float4*)(&dst_v[row * D_PAD + col_half]) = make_float4(0,0,0,0);
        }

        pipe.producer_commit();
    }

    // --------------------------------------------------------
    // MAIN LOOP
    // --------------------------------------------------------
    for (int k_idx = 0; k_idx < total_k_blocks; ++k_idx) {
        // Trigger Next Load (if exists)
        int next_k_idx = k_idx + 1;
        if (next_k_idx < total_k_blocks) {
             int next_write_stage = write_stage ^ 1;
             int k_row_start = next_k_idx * 16;

             pipe.producer_acquire();

             __half* dst_k = s_K_base + next_write_stage * stage_stride;
             __half* dst_v = s_V_base + next_write_stage * stage_stride;
             const __half* src_k = k_ptr + k_row_start * D;
             const __half* src_v = v_ptr + k_row_start * D;

             int idx = tid;
             int row = idx / 8;
             int col_half = (idx % 8) * 8;
             int src_offset_half = idx * 8;

             if (k_row_start * D + src_offset_half < N * D) {
                cuda::memcpy_async(&dst_k[row * D_PAD + col_half], &src_k[src_offset_half], 16, pipe);
                cuda::memcpy_async(&dst_v[row * D_PAD + col_half], &src_v[src_offset_half], 16, pipe);
             } else {
                 *(float4*)(&dst_k[row * D_PAD + col_half]) = make_float4(0,0,0,0);
                 *(float4*)(&dst_v[row * D_PAD + col_half]) = make_float4(0,0,0,0);
             }

             pipe.producer_commit();
        }

        // Wait for Current Load
        // We need data at 'read_stage'.
        // This corresponds to the oldest commit.
        // If we are at step k, we have (k+1) commits pending (including prologue).
        // No, commit logic:
        // Prologue: Commit 0.
        // Loop 0: Issue Commit 1. Wait for Commit 0.
        // Loop 1: Issue Commit 2. Wait for Commit 1.
        pipe.consumer_wait();

        // Sync threads to ensure all threads see the shared memory update?
        // cuda::pipeline implies visibility?
        // "Arriving at the barrier guarantees that the data is visible".
        // But we need __syncthreads() for logic coherence across warps if we rely on it?
        // consumer_wait is a block-wide sync if called by all threads?
        // "cuda::pipeline<cuda::thread_scope_thread>" means scope is thread.
        // So each thread waits for its OWN copy.
        // But we need the whole block to be ready.
        // So we need __syncthreads() after consumer_wait.
        __syncthreads();

        // COMPUTE with read_stage
        __half* curr_s_K = s_K_base + read_stage * stage_stride;
        __half* curr_s_V = s_V_base + read_stage * stage_stride;

        // ... WMMA Logic (Same as before) ...
        int my_row_offset = warpId * 16;

        // 1. S = Q * K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);

        for (int d_chunk = 0; d_chunk < 4; ++d_chunk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, s_Q + my_row_offset * D_PAD + d_chunk * 16, D_PAD);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, curr_s_K + d_chunk * 16, D_PAD); // Use curr_s_K

            wmma::mma_sync(acc_s, a_frag, b_frag, acc_s);
        }

        // 2. Softmax
        wmma::store_matrix_sync(s_S + my_row_offset * 16, acc_s, 16, wmma::mem_row_major);

        // __syncwarp is sufficient for intra-warp dependency
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
                s_P[row_abs * 16 + c] = __float2half(e);
                row_sum += e;
            }

            s_l[row_abs] = s_l[row_abs] * alpha + row_sum;
        }

        // 2c. Rescale Acc_O
        for(int i=0; i<4; ++i) {
             wmma::store_matrix_sync(s_O_scratch + my_row_offset * D_PAD + i*16, acc_o[i], D_PAD, wmma::mem_row_major);
        }

        for (int k = 0; k < 32; ++k) {
             int idx = k * 32 + laneId;
             int r_rel = idx / 64;
             int c = idx % 64;
             int r_abs = my_row_offset + r_rel;
             float alpha = s_alpha[r_abs];
             s_O_scratch[r_abs * D_PAD + c] *= alpha;
        }

        for(int i=0; i<4; ++i) {
             wmma::load_matrix_sync(acc_o[i], s_O_scratch + my_row_offset * D_PAD + i*16, D_PAD, wmma::mem_row_major);
        }

        // 3. O += P * V
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, s_P + my_row_offset * 16, 16);

        for (int i = 0; i < 4; ++i) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, curr_s_V + i*16, D_PAD); // Use curr_s_V
            wmma::mma_sync(acc_o[i], p_frag, v_frag, acc_o[i]);
        }

        // Release Consumer
        pipe.consumer_release();

        // Wait for threads before switching pointers?
        // We need to ensure everyone is done with curr_s_K before we overwrite it (in future).
        // But pipeline ensures we don't overwrite until we produce again.
        // Since buffer is size 2. We produce to `next`. We consume `curr`.
        // `next` != `curr`.
        // So we are safe.
        // Only need block sync at end of compute to synchronize Warps?
        // We have __syncthreads() at start of loop (after wait).
        // So next loop won't start compute until all threads passed wait.
        // It's correct.

        read_stage ^= 1;
        write_stage ^= 1;
    }

    // Write Output
    for(int i=0; i<4; ++i) {
         wmma::store_matrix_sync(s_O_scratch + (warpId * 16) * D_PAD + i*16, acc_o[i], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int k = 0; k < 32; ++k) {
         int idx = tid + k * 128;
         if (idx < Br * D) {
             int r = idx / D;
             int c = idx % D;
             float val = s_O_scratch[r * D_PAD + c];
             int global_row = q_row_start + r;
             if (global_row < N) {
                 float l = s_l[r];
                 o_ptr[global_row * D + c] = __float2half(val / (l + 1e-6f));
                 if (c == 0) {
                     l_ptr[global_row] = s_m[r] + logf(l + 1e-6f);
                 }
             }
         }
    }
}

// Host wrapper update
torch::Tensor flash_attn_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);

    auto O = torch::zeros_like(Q);
    auto L = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    // Br=64
    int grid_n = (N + 64 - 1) / 64;
    dim3 grid(grid_n, B * H);
    dim3 block(128);

    // Shared mem: ~37KB + overhead.
    // Make sure to request enough.
    // D_PAD=72. Br=64. Bc=16.
    // Q: 64*72*2 = 9216.
    // K, V: 2 * 16 * 72 * 2 = 4608 each. Total 9216.
    // O_scratch: 64*72*4 = 18432.
    // Stats: ~5KB.
    // Total ~ 42KB.
    int shared_mem_size = 48 * 1024; // Request 48KB

    // Check arch? This requires sm_70+.
    // flash_attn_wmma_pipeline is called.

    flash_attn_wmma_pipeline<<<grid, block, shared_mem_size>>>(
        (const __half*)Q.data_ptr(),
        (const __half*)K.data_ptr(),
        (const __half*)V.data_ptr(),
        (__half*)O.data_ptr(),
        (float*)L.data_ptr(),
        softmax_scale,
        N, 0, 0, 0
    );

    return O;
}
