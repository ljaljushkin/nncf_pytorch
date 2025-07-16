import sys

import torch

sys.path.append("/home/nlyaly/projects/nncf")

# Test the simple implementation with the exact same test case from gemma_code.py
from gemma_code import sum_like_v2_two_stage_simple


def test_simple_implementation():
    print("Testing sum_like_v2_two_stage_simple...")

    # Use the exact same test case
    tensor_to_sum = torch.randn(4, 16, 16, 16, device="cuda", dtype=torch.float16)
    ref_tensor = torch.empty(1, 16, 1, 1, device="cuda", dtype=torch.float16)

    # PyTorch reference
    pytorch_result = torch.sum(tensor_to_sum, axis=(0, 2, 3), keepdim=True)
    print(f"PyTorch result: {pytorch_result.shape}, dtype: {pytorch_result.dtype}")

    # Our implementation
    triton_result = sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor)
    print(f"Triton result: {triton_result.shape}, dtype: {triton_result.dtype}")

    # Check correctness
    correct = torch.allclose(pytorch_result, triton_result, rtol=1, atol=1e-1)
    print(f"Correct: {correct}")

    if not correct:
        print(f"Max diff: {(pytorch_result - triton_result).abs().max()}")
        print(f"First few PyTorch values: {pytorch_result.flatten()[:5]}")
        print(f"First few Triton values: {triton_result.flatten()[:5]}")

    # Test performance
    import triton.testing

    print("\nBenchmarking...")
    pytorch_time = triton.testing.do_bench(lambda: torch.sum(tensor_to_sum, axis=(0, 2, 3), keepdim=True))

    triton_time = triton.testing.do_bench(lambda: sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor))

    print(f"PyTorch time: {pytorch_time:.4f}ms")
    print(f"Triton time: {triton_time:.4f}ms")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")


if __name__ == "__main__":
    test_simple_implementation()
