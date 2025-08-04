#!/usr/bin/env python3
"""
Simple test script for the optimized per-group CUDA kernel
"""

import sys
import time

import torch
import torch.nn.functional as F


def test_optimized_kernel():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        return

    print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Test basic CUDA operations first
    try:
        x = torch.randn(100).cuda()
        y = x * 2
        print("✓ Basic CUDA operations work")
    except Exception as e:
        print(f"✗ Basic CUDA operations failed: {e}")
        return

    # Now test the NNCF import
    try:
        sys.path.append("/home/nlyaly/projects/nncf2/tools")
        from benchmark_quantize_layers import BatchDescriptor
        from benchmark_quantize_layers import BatchMode
        from benchmark_quantize_layers import ExecutionType
        from benchmark_quantize_layers import GranularityType
        from benchmark_quantize_layers import ParamStruct
        from benchmark_quantize_layers import TensorType
        from benchmark_quantize_layers import TimingMode
        from benchmark_quantize_layers import get_module

        print("✓ NNCF import successful")
    except Exception as e:
        print(f"✗ NNCF import failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test the optimized kernel with the problematic shape
    try:
        print("\nTesting optimized per-group kernel...")

        # Create test parameters for per-group quantization [2048, 128256]
        batch_desc = BatchDescriptor(
            mode=BatchMode.HIGH, input_size=[2048, 128256], num_runs={torch.device("cuda"): 100}
        )

        param_struct = ParamStruct(
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            exec_type=ExecutionType.REGULAR,
            batch=batch_desc,
            tensor_type=TensorType.WEIGHTS,
            granularity=GranularityType.PER_GROUP,
            symmetric=False,
            narrow_range=False,
            timing_mode=TimingMode.KERNEL,
            ref=False,
        )

        # Create quantizer using get_module
        quantizer = get_module(param_struct)

        # Create input tensor
        input_tensor = torch.randn(2048, 128256, requires_grad=True, device="cuda", dtype=torch.bfloat16)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Input elements: {input_tensor.numel():,}")

        # Forward pass
        start_time = time.time()
        output = quantizer(input_tensor)
        forward_time = time.time() - start_time

        print(f"✓ Forward pass successful: {forward_time:.4f}s")
        print(f"Output shape: {output.shape}")

        # Backward pass - this is where the optimized kernel should be used
        start_time = time.time()
        loss = output.sum()
        loss.backward()
        backward_time = time.time() - start_time

        print(f"✓ Backward pass successful: {backward_time:.4f}s")
        print(f"Gradient computed: {input_tensor.grad is not None}")

        if input_tensor.grad is not None:
            print(f"Gradient shape: {input_tensor.grad.shape}")
            print(f"Gradient mean: {input_tensor.grad.mean().item():.6f}")
            print(f"Gradient std: {input_tensor.grad.std().item():.6f}")

        print(f"\n🎉 SUCCESS! Optimized kernel test completed")
        print(f"Total time: {forward_time + backward_time:.4f}s")
        print(f"Backward time (optimized kernel): {backward_time:.4f}s")

    except Exception as e:
        print(f"✗ Kernel test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_optimized_kernel()
