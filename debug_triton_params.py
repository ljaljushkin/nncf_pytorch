#!/usr/bin/env python3

import numpy as np
import torch


def analyze_tensor_parameters():
    """Analyze the parameters for [2048, 128256] tensor to understand Triton's approach"""

    # Create test tensors matching our benchmark
    input_tensor = torch.randn(2048, 128256, dtype=torch.bfloat16, device="cuda")

    # For per-channel quantization, typical parameter shapes:
    # Per-weight-channel: params [2048, 1, ...]
    # Per-activation-channel: params [1, 128256] (less common)

    print("=== Tensor Shape Analysis ===")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Total elements: {input_tensor.numel():,}")

    # Test different parameter configurations
    configs = [
        ("Per-weight-channel", (2048, 1), input_tensor.numel() // 2048),
        ("Per-activation-channel", (1, 128256), input_tensor.numel() // 128256),
        ("Groups of 128", (2048 * 128256 // 128,), 128),  # What Triton seems to use
        ("Groups of 1002", (2048 * 1002,), 128),  # Based on compiled kernel shape
    ]

    for name, param_shape, elements_per_scale in configs:
        print(f"\n{name}:")
        print(f"  Parameter shape: {param_shape}")
        print(f"  Elements per scale: {elements_per_scale:,}")
        print(f"  Scale count: {input_tensor.numel() // elements_per_scale:,}")

        # Check if this matches Triton's x1 = xindex // 128 pattern
        if elements_per_scale == 128:
            print(f"  ✅ MATCHES Triton's pattern (x1 = xindex // 128)")

    # From the compiled kernel we see:
    # arg1_1 = rand_strided((2048, 1002, 1), (1002, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    # arg2_1 = rand_strided((2048, 1002, 1), (1002, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    print(f"\n=== Compiled Triton Kernel Analysis ===")
    print(f"Input: [2048, 128256] -> reshaped to [2048, 1002, 128]")
    print(f"Parameters: [2048, 1002, 1] with stride (1002, 1, 1)")
    print(f"Elements per scale: 128 (matches x1 = xindex // 128)")
    print(f"This creates 2048 * 1002 = 2,052,096 scales")
    print(f"Each scale handles 128 elements")

    return 128  # Elements per scale that Triton uses


if __name__ == "__main__":
    analyze_tensor_parameters()
