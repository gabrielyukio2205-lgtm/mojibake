# Simple FlashAttention Implementation (WMMA)

This is a **Tensor Core-accelerated** implementation of FlashAttention (Forward Pass) written in CUDA C++ using `nvcuda::wmma`.
It implements the tiling and online softmax algorithms described in the FlashAttention papers, utilizing hardware matrix multiplication units.

## Key Features

*   **WMMA (Warp Matrix Multiply Accumulate):** Uses NVIDIA Tensor Cores for high-performance GEMM operations (`16x16x16` tiles).
*   **Warp-Level Parallelism:** Each warp processes a 16x64 block of Q.
*   **Online Softmax:** Fused softmax implementation to minimize memory I/O.
*   **FP16/FP32 Mixed Precision:** Inputs are FP16, accumulation is FP32 (Tensor Core standard).

## Prerequisites

*   NVIDIA GPU with Tensor Cores (Volta, Turing, Ampere, Hopper).
*   Compute Capability 7.5+.
*   CUDA Toolkit.
*   PyTorch.

## Installation

```bash
python setup.py install
```

## Usage

```python
import torch
import simple_flash_attn_cuda

# Dimensions: [Batch, Heads, SeqLen, HeadDim]
# HeadDim (D) should be 64.
q = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
softmax_scale = 1.0 / 8.0

# Run
output = simple_flash_attn_cuda.forward(q, k, v, softmax_scale)
```
