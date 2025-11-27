import torch
import math
import sys
import os

# Try to import the extension
try:
    import simple_flash_attn_cuda
    print("Extension loaded successfully.")
except ImportError:
    print("Extension not installed. This script is for demonstration purposes.")
    print("To run this, you need to build the extension first:")
    print("  python setup.py install")
    sys.exit(0)

def manual_attention(q, k, v):
    # q, k, v: [B, H, N, D]
    d = q.size(-1)
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out

def run_test():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping test.")
        return

    device = "cuda"
    dtype = torch.float16

    B = 2
    H = 4
    N = 128
    D = 64

    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    scale = 1.0 / math.sqrt(D)

    print(f"Running FlashAttention with B={B}, H={H}, N={N}, D={D}")

    # Run custom kernel
    # Note: Our C++ extension exposes 'forward'
    out_cuda = simple_flash_attn_cuda.forward(q, k, v, scale)

    # Run reference
    out_ref = manual_attention(q.float(), k.float(), v.float()).to(dtype)

    # Or use torch implementation if available (PyTorch 2.0+)
    # out_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Compare
    diff = (out_cuda - out_ref).abs().max().item()
    print(f"Max difference: {diff}")

    if diff < 1e-2:
        print("Test PASSED!")
    else:
        print("Test FAILED!")

if __name__ == "__main__":
    run_test()
