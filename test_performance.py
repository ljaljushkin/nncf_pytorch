import sys

import torch

sys.path.append("/home/nlyaly/projects/nncf")
from gemma_code import sum_like_v2_two_stage_simple


# Test performance comparison
def test_performance():
    print("Performance comparison for sum_like_v2_two_stage_simple:")

    # Test with different sizes
    sizes = [
        (4, 16, 16, 16),
        (4, 32, 32, 32),
        (4, 64, 64, 64),
        (128, 64, 32, 16),
    ]

    for shape in sizes:
        tensor_to_sum = torch.randn(shape, device="cuda", dtype=torch.float16)
        ref_tensor = torch.empty(1, shape[1], 1, 1, device="cuda", dtype=torch.float16)

        # PyTorch reference
        pytorch_result = torch.sum(tensor_to_sum, axis=(0, 2, 3), keepdim=True)

        # Our implementation
        triton_result = sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor)

        # Check correctness
        correct = torch.allclose(pytorch_result, triton_result, rtol=1e-2, atol=1e-2)

        print(f"Shape {shape}: Correct={correct}")
        if not correct:
            print(f"  Max diff: {(pytorch_result - triton_result).abs().max()}")

        # Performance test
        import time

        # Warmup
        for _ in range(10):
            torch.sum(tensor_to_sum, axis=(0, 2, 3), keepdim=True)
            sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor)

        # Time PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            torch.sum(tensor_to_sum, axis=(0, 2, 3), keepdim=True)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start

        # Time Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor)
        torch.cuda.synchronize()
        triton_time = time.time() - start

        speedup = pytorch_time / triton_time
        print(f"  PyTorch: {pytorch_time * 1000:.2f}ms, Triton: {triton_time * 1000:.2f}ms, Speedup: {speedup:.2f}x")
        print()


if __name__ == "__main__":
    test_performance()
