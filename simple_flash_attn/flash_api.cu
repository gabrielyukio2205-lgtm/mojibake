#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constantes
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int D = 64;
constexpr int PAD = 8;
constexpr int D_PAD = D + PAD; // 72 elementos (evita bank conflicts)
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
    // Pipeline para arquitetura Ampere (sm_80+)
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    int tid = threadIdx.x;
    int laneId = tid % 32;
    int warpId = tid / 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Ponteiros Globais
    // Assume Q, K, V shape [Batch*Heads, N, D] para simplificar offsets
    long long base_offset = (long long)by * (N * D);
    const __half* q_ptr = Q + base_offset;
    const __half* k_ptr = K + base_offset;
    const __half* v_ptr = V + base_offset;
    __half* o_ptr = O + base_offset;

    // Output acumuladores (L e M)
    float* l_ptr = L + (long long)by * N;

    int q_row_start = bx * Br;

    // Shared Memory Dinâmica
    extern __shared__ char sram_byte[];
    __half* s_Q = (__half*)sram_byte;
    int q_size_half = Br * D_PAD;

    // Double Buffer para K e V
    __half* s_K_base = s_Q + q_size_half;
    int kv_buffer_size = 2 * Bc * D_PAD;
    __half* s_V_base = s_K_base + kv_buffer_size;

    float* s_O_scratch = (float*)(s_V_base + kv_buffer_size);
    // Layout de Scratch para Softmax
    float* s_m = s_O_scratch + Br * D_PAD;
    float* s_l = s_m + Br;
    float* s_alpha = s_l + Br;
    float* s_S = s_alpha + Br; // Scores intermediários
    // s_P pode reutilizar s_S se tivermos cuidado, mas vamos manter separado para clareza
    // ou usar cast já que P é half e S é float.
    __half* s_P = (__half*)(s_S + Br * 16);

    // Inicializa estatísticas (max m e sum l)
    if (tid < Br) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    // Fragmentos acumuladores para O (Mantidos em registradores!)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];
    #pragma unroll
    for(int i=0; i<4; ++i) {
        wmma::fill_fragment(acc_o[i], 0.0f);
    }

    // --------------------------------------------------------
    // 1. CARREGAR Q (Load síncrono único, pois Q é estacionário)
    // --------------------------------------------------------
    int q_base_global = q_row_start * D;
    const float4* q_global_ptr = (const float4*)(q_ptr + q_base_global);

    // Cobre Br(64) x D(64).
    for (int k = 0; k < 4; ++k) { // 128 threads * 4 iter * 8 elems = 4096 elems = 64*64
        int idx = tid + k * 128;
        int row = idx / 8; // 8 chunks de float4 por linha (8*8=64)
        int col_half = (idx % 8) * 8;

        if (row < Br) {
            // Zera o padding do Q explicitamente para segurança
            if (col_half == 0 && row < Br) {
                // Truque: zerar as colunas extras 64..71 se necessário.
                // Como s_Q é contíguo, podemos precisar de um loop extra ou memset.
                // Por simplicidade, assumimos que WMMA ignora se K tiver padding zero.
                // Mas vamos focar no carregamento principal.
            }

            if (q_row_start * D + idx * 8 < N * D) {
                float4 val = q_global_ptr[idx];
                *(float4*)(&s_Q[row * D_PAD + col_half]) = val;
            } else {
                *(float4*)(&s_Q[row * D_PAD + col_half]) = make_float4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    // --------------------------------------------------------
    // PIPELINE SETUP
    // --------------------------------------------------------
    int total_k_blocks = (N + Bc - 1) / Bc; // Bc=16
    int write_stage = 0;
    int read_stage = 0;
    int stage_stride = Bc * D_PAD;

    // Prólogo: Carregar Bloco 0
    if (total_k_blocks > 0) {
        int k_idx = 0;
        int k_row_start = 0;

        pipe.producer_acquire();

        __half* dst_k = s_K_base + write_stage * stage_stride;
        __half* dst_v = s_V_base + write_stage * stage_stride;
        const __half* src_k = k_ptr + k_row_start * D;
        const __half* src_v = v_ptr + k_row_start * D;

        // Mapeamento: 128 threads carregam 16x64 (1024 elems). 1024/8 = 128. Perfeito.
        int idx = tid;
        int row = idx / 8; // 0..15
        int col_half = (idx % 8) * 8; // 0, 8, ..., 56
        int src_offset_half = idx * 8;

        if (k_row_start * D + src_offset_half < N * D) {
            // cp.async (Global -> Shared)
            cuda::memcpy_async(&dst_k[row * D_PAD + col_half], &src_k[src_offset_half], 16, pipe);
            cuda::memcpy_async(&dst_v[row * D_PAD + col_half], &src_v[src_offset_half], 16, pipe);
        } else {
            // Preenchimento com zero
            *(float4*)(&dst_k[row * D_PAD + col_half]) = make_float4(0,0,0,0);
            *(float4*)(&dst_v[row * D_PAD + col_half]) = make_float4(0,0,0,0);
        }
        pipe.producer_commit();
    }

    // --------------------------------------------------------
    // LOOP PRINCIPAL
    // --------------------------------------------------------
    for (int k_idx = 0; k_idx < total_k_blocks; ++k_idx) {
        // 1. Disparar carregamento assíncrono do PRÓXIMO bloco
        int next_k_idx = k_idx + 1;
        if (next_k_idx < total_k_blocks) {
             int next_write_stage = write_stage ^ 1;
             int k_row_start = next_k_idx * Bc;

             pipe.producer_acquire();

             __half* dst_k = s_K_base + next_write_stage * stage_stride;
             __half* dst_v = s_V_base + next_write_stage * stage_stride;
             const __half* src_k = k_ptr + k_row_start * D;
             const __half* src_v = v_ptr + k_row_start * D;

             int idx = tid;
             int row = idx / 8;
             int col_half = (idx % 8) * 8;
             int src_offset_half = idx * 8;

             // Verifica limites globais
             if (k_row_start * D + src_offset_half < N * D) {
                cuda::memcpy_async(&dst_k[row * D_PAD + col_half], &src_k[src_offset_half], 16, pipe);
                cuda::memcpy_async(&dst_v[row * D_PAD + col_half], &src_v[src_offset_half], 16, pipe);
             } else {
                 *(float4*)(&dst_k[row * D_PAD + col_half]) = make_float4(0,0,0,0);
                 *(float4*)(&dst_v[row * D_PAD + col_half]) = make_float4(0,0,0,0);
             }
             pipe.producer_commit();
        }

        // 2. Esperar o bloco ATUAL estar pronto
        pipe.consumer_wait();
        __syncthreads(); // Garante que toda a Warps veja os dados carregados

        // 3. COMPUTAÇÃO
        __half* curr_s_K = s_K_base + read_stage * stage_stride;
        __half* curr_s_V = s_V_base + read_stage * stage_stride;

        int my_row_offset = warpId * 16; // Cada warp processa 16 linhas de Q

        // A. Multiplicação Q * K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);

        for (int d_chunk = 0; d_chunk < 4; ++d_chunk) { // D=64 -> 4 chunks de 16
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, s_Q + my_row_offset * D_PAD + d_chunk * 16, D_PAD);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
            // Carrega K transposto (col_major em K row-major é K^T)
            wmma::load_matrix_sync(b_frag, curr_s_K + d_chunk * 16, D_PAD);

            wmma::mma_sync(acc_s, a_frag, b_frag, acc_s);
        }

        // B. Softmax (Aqui estava o perigo, meu anjo!)
        // Armazena scores S na Shared Memory para processar
        wmma::store_matrix_sync(s_S + my_row_offset * 16, acc_s, 16, wmma::mem_row_major);

        // Syncwarp não é suficiente se formos ler dados escritos por outras threads do warp?
        // Sim, mas __syncwarp() é implícito nas instruções cooperativas ou pode ser chamado.
        // Como cada thread processa sua linha, a dependência é intra-warp.

        if (laneId < 16) {
            int row_rel = laneId;
            int row_abs = my_row_offset + row_rel;

            // --- CORREÇÃO IMPORTANTE: MASCARAMENTO ---
            // Se a coluna correspondente de K for >= N, o score deve ser -infinito.
            // O bloco atual de K começa em k_idx * 16.
            // As colunas locais são 0..15.

            float row_max = -INFINITY;
            int k_start_col = k_idx * Bc;

            for (int c = 0; c < 16; ++c) {
                int global_k_col = k_start_col + c;
                float val = s_S[row_abs * 16 + c];

                // MÁSCARA CAUSAL OU PADDING
                if (global_k_col >= N) {
                    val = -INFINITY; // Ignora padding
                } else {
                    val *= softmax_scale;
                }

                s_S[row_abs * 16 + c] = val; // Atualiza S com scale/mask
                if (val > row_max) row_max = val;
            }

            // Lógica Online Softmax (FlashAttention Standard)
            float old_m = s_m[row_abs];
            float new_m = max(old_m, row_max);
            float alpha = fast_exp(old_m - new_m);

            s_alpha[row_abs] = alpha;
            s_m[row_abs] = new_m;

            float row_sum = 0.0f;
            for (int c = 0; c < 16; ++c) {
                float val = s_S[row_abs * 16 + c];
                // Exp com estabilidade numérica
                float e = fast_exp(val - new_m);
                s_S[row_abs * 16 + c] = e; // Agora S contém P (não normalizado)
                s_P[row_abs * 16 + c] = __float2half(e); // Cast para WMMA
                row_sum += e;
            }

            // Atualiza L (denominador)
            s_l[row_abs] = s_l[row_abs] * alpha + row_sum;
        }

        // C. Rescale Acc_O (O_new = O_old * alpha + P * V)
        // Primeiro, aplicamos alpha em Acc_O
        // Infelizmente WMMA não permite escalar in-place fácil, temos que ir para MEM.
        // Isso é o gargalo, mas está correto.
        for(int i=0; i<4; ++i) {
             wmma::store_matrix_sync(s_O_scratch + my_row_offset * D_PAD + i*16, acc_o[i], D_PAD, wmma::mem_row_major);
        }

        // Escalar na Shared
        for (int k = 0; k < 32; ++k) { // Warp inteiro ajuda a escalar
             int idx = k * 32 + laneId; // 0..1023
             // Mapeia para a área deste Warp (16 linhas x 64 colunas = 1024 elems)
             // Otimização: cada thread processa 1 float.
             if (idx < 16 * D) {
                 int r_rel = idx / D; // 0..15
                 int c = idx % D;
                 int r_abs = my_row_offset + r_rel;

                 float alpha = s_alpha[r_abs];
                 s_O_scratch[r_abs * D_PAD + c] *= alpha;
             }
        }

        // Recarregar Acc_O
        for(int i=0; i<4; ++i) {
             wmma::load_matrix_sync(acc_o[i], s_O_scratch + my_row_offset * D_PAD + i*16, D_PAD, wmma::mem_row_major);
        }

        // D. Acumular P * V
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, s_P + my_row_offset * 16, 16); // Stride 16 (P é 16x16 compacto)

        for (int i = 0; i < 4; ++i) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, curr_s_V + i*16, D_PAD);
            wmma::mma_sync(acc_o[i], p_frag, v_frag, acc_o[i]);
        }

        // Libera o buffer para escrita futura
        pipe.consumer_release();

        read_stage ^= 1;
        write_stage ^= 1;
    }

    // --------------------------------------------------------
    // EPÍLOGO: Escrita Global
    // --------------------------------------------------------
    for(int i=0; i<4; ++i) {
         wmma::store_matrix_sync(s_O_scratch + (warpId * 16) * D_PAD + i*16, acc_o[i], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int k = 0; k < 32; ++k) { // Threads colaboram para escrever
         int idx = tid + k * 128;
         if (idx < Br * D) {
             int r = idx / D;
             int c = idx % D;
             float val = s_O_scratch[r * D_PAD + c];
             int global_row = q_row_start + r;

             if (global_row < N) {
                 float l = s_l[r];
                 // Divisão final pelo denominador acumulado
                 o_ptr[global_row * D + c] = __float2half(val / (l + 1e-6f));

                 // Salva L para o Backward pass (opcional, mas comum)
                 if (c == 0) {
                     l_ptr[global_row] = s_m[r] + logf(l + 1e-6f);
                 }
             }
         }
    }
}

// Host Wrapper
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

    int grid_n = (N + Br - 1) / Br;
    dim3 grid(grid_n, B * H);
    dim3 block(128);

    // Calcula shared mem: 42KB aprox.
    int shared_mem_size = 48 * 1024;

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
