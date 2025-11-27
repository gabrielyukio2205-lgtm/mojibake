# Simple FlashAttention Implementation

This is a simplified but high-performance implementation of FlashAttention (Forward Pass) written in CUDA C++.
It implements the tiling and online softmax algorithms described in the FlashAttention papers.

**Note:** This implementation targets `D=64` (Head Dimension) and uses `FP16`.

## Prerequisites

*   NVIDIA GPU
*   CUDA Toolkit (11.0+)
*   PyTorch

## Installation

```bash
python setup.py install
```

## Usage

```python
import torch
import simple_flash_attn_cuda

# Dimensions: [Batch, Heads, SeqLen, HeadDim]
q = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4, 128, 64, device='cuda', dtype=torch.float16)
softmax_scale = 1.0 / 8.0

# Run
output = simple_flash_attn_cuda.forward(q, k, v, softmax_scale)
```

## Performance Notes

This kernel uses:
*   **Tiling:** Loads blocks of Q, K, V into Shared Memory (SRAM) to reduce HBM access.
*   **Online Softmax:** Computes softmax incrementally to avoid materializing the large attention matrix.
*   **Vectorization:** Uses `float4` (128-bit) memory accesses for maximum bandwidth.
*   **Register Accumulation:** Accumulates output in registers.

It is comparable to FlashAttention-2 in design principles. FlashAttention-3 introduces specific optimizations for Hopper GPUs (H100) such as Warp-Group MMA and TMA (Tensor Memory Accelerator), which are not included here to maintain compatibility and simplicity.
