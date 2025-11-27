#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor flash_attn_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale
);

// C++ interface
torch::Tensor flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale) {

    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    return flash_attn_cuda_forward(Q, K, V, softmax_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "FlashAttention Forward (CUDA)");
}
