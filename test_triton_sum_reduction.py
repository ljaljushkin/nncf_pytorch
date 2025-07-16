#!/usr/bin/env python3
"""
Comprehensive test for Triton sum reduction using the same configurations as test_quantize_asymmetric_backward.

This test covers:
- per_channel vs single_scale (per_tensor)
- weights vs activations
- different input sizes
- different block sizes
- fp16 vs fp32
- CUDA vs CPU
"""

import pytest
import torch

from nncf.torch.quantization.triton.reference import triton_sum_like


def idfn(val):
    """ID function for pytest parametrization"""
    if isinstance(val, list):
        return "[{}]".format("-".join([str(v) for v in val]))
    return None


def generate_reference_tensor(input_size, scale_mode, is_weights, dtype):
    """Generate reference tensor shape based on scale mode and tensor type"""
    if scale_mode == "single_scale":
        # Single scale: [1]
        return torch.ones([1], dtype=dtype)
    elif scale_mode == "per_channel_scale":
        if is_weights:
            # For weights, channel is dim 0: [C, 1, 1, 1]
            channel_count = input_size[0]
            if channel_count == 1:
                pytest.skip("Same case as for single scale mode")
            ref_shape = [1 for _ in input_size]
            ref_shape[0] = channel_count
        else:
            # For activations, channel is dim 1: [1, C, 1, 1]
            channel_count = input_size[1]
            if channel_count == 1:
                pytest.skip("Same case as for single scale mode")
            ref_shape = [1 for _ in input_size]
            ref_shape[1] = channel_count
        return torch.ones(ref_shape, dtype=dtype)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")


def pytorch_sum_like(tensor_to_sum, ref_tensor):
    """PyTorch reference implementation of sum_like functionality"""
    # Ensure both tensors have the same number of dimensions
    while tensor_to_sum.dim() < ref_tensor.dim():
        tensor_to_sum = tensor_to_sum.unsqueeze(0)
    while ref_tensor.dim() < tensor_to_sum.dim():
        ref_tensor = ref_tensor.unsqueeze(0)

    # Sum over dimensions where ref_tensor has size 1
    result = tensor_to_sum
    for dim in range(result.dim()):
        if ref_tensor.size(dim) == 1 and result.size(dim) > 1:
            result = result.sum(dim, keepdim=True)

    return result


# def skip_if_half_on_cpu(is_fp16, use_cuda):
#     """Skip test if trying to use fp16 on CPU"""
#     if is_fp16 and not use_cuda:
#         pytest.skip("FP16 not supported on CPU")


@pytest.mark.parametrize("is_weights", [True, False], ids=["weights", "activations"])
# @pytest.mark.parametrize("scale_mode", ["single_scale", "per_channel_scale"])
@pytest.mark.parametrize(
    "input_size",
    [[1, 16, 64, 64], [4, 16, 16, 16], [1024, 256], [8, 256, 32, 32], [4096, 4096], [256, 1], [4, 64, 128, 128]],
    ids=idfn,
)
# @pytest.mark.parametrize("input_size", [[2, 2, 2, 2]], ids=idfn)
# @pytest.mark.parametrize("block_size", [8, 32, 256, 512, 1024], ids=["bs8", "bs32", "bs256", "bs512", "bs1024"])
# @pytest.mark.parametrize("block_size", [2, 8, 16], ids=["bs2", "bs8", "bs16"])
class TestTritonSumReduction:
    def test_triton_sum_like_correctness(self, is_weights, input_size):
        use_cuda = True
        is_fp16 = True
        scale_mode = "per_channel_scale"

        device = torch.device("cuda" if use_cuda else "cpu")
        dtype = torch.float16 if is_fp16 else torch.float32

        # Generate test data
        torch.manual_seed(42)

        # input_tensor = torch.arange(math.prod(input_size), device=device, dtype=dtype).reshape(input_size)
        input_tensor = torch.randn(input_size, device=device, dtype=dtype)
        ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

        # Compute expected result using PyTorch
        expected = pytorch_sum_like(input_tensor, ref_tensor)

        # Compute result using triton_sum_like
        result = triton_sum_like(input_tensor, ref_tensor, block_size=256)

        # Check results
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        rtol = 1 if is_fp16 else 1e-5
        atol = 1e-1 if is_fp16 else 1e-6

        assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
            f"Results don't match. Max diff: {(result - expected).abs().max()}"
        )

    # def test_optimized_kernel_correctness(self, use_cuda, is_weights, scale_mode, is_fp16, input_size, block_size):
    #     """Test that optimized_sum_reduction_kernel produces correct results"""
    #     if not torch.cuda.is_available() and use_cuda:
    #         pytest.skip("Skipping CUDA test cases for CPU only setups")

    #     skip_if_half_on_cpu(is_fp16, use_cuda)

    #     # Skip CPU tests since Triton requires CUDA
    #     if not use_cuda:
    #         pytest.skip("Triton requires CUDA")

    #     device = torch.device("cuda" if use_cuda else "cpu")
    #     dtype = torch.float16 if is_fp16 else torch.float32

    #     # Generate test data
    #     torch.manual_seed(42)
    #     input_tensor = torch.randn(input_size, device=device, dtype=dtype)
    #     ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

    #     # Compute expected result using PyTorch
    #     expected = pytorch_sum_like(input_tensor, ref_tensor)

    #     # Compute result using optimized kernel directly
    #     output = torch.zeros_like(ref_tensor)
    #     input_meta = get_4d_tensor_meta(input_tensor)
    #     output_meta = get_4d_tensor_meta(output)

    #     import triton

    #     grid_size = triton.cdiv(input_tensor.numel(), block_size)

    #     optimized_sum_reduction_kernel[(grid_size,)](
    #         input_tensor,
    #         input_meta,
    #         output,
    #         output_meta,
    #         BLOCK_SIZE=block_size,
    #     )

    #     # Check results
    #     assert output.shape == expected.shape, f"Shape mismatch: {output.shape} vs {expected.shape}"
    #     rtol = 1e-2 if is_fp16 else 1e-5
    #     atol = 1e-3 if is_fp16 else 1e-6

    #     assert torch.allclose(output, expected, rtol=rtol, atol=atol), (
    #         f"Results don't match. Max diff: {(output - expected).abs().max()}"
    #     )

    # def test_working_optimized_kernel_correctness(
    #     self, use_cuda, is_weights, scale_mode, is_fp16, input_size, block_size
    # ):
    #     """Test that working_optimized_sum_reduction_kernel produces correct results"""
    #     if not torch.cuda.is_available() and use_cuda:
    #         pytest.skip("Skipping CUDA test cases for CPU only setups")

    #     skip_if_half_on_cpu(is_fp16, use_cuda)

    #     # Skip CPU tests since Triton requires CUDA
    #     if not use_cuda:
    #         pytest.skip("Triton requires CUDA")

    #     device = torch.device("cuda" if use_cuda else "cpu")
    #     dtype = torch.float16 if is_fp16 else torch.float32

    #     # Generate test data
    #     torch.manual_seed(42)
    #     input_tensor = torch.randn(input_size, device=device, dtype=dtype)
    #     ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

    #     # Compute expected result using PyTorch
    #     expected = pytorch_sum_like(input_tensor, ref_tensor)

    #     # Compute result using working optimized kernel directly
    #     output = torch.zeros_like(ref_tensor)
    #     input_meta = get_4d_tensor_meta(input_tensor)
    #     output_meta = get_4d_tensor_meta(output)

    #     import triton

    #     grid_size = triton.cdiv(input_tensor.numel(), block_size)

    #     working_optimized_sum_reduction_kernel[(grid_size,)](
    #         input_tensor,
    #         input_meta,
    #         output,
    #         output_meta,
    #         BLOCK_SIZE=block_size,
    #     )

    #     # Check results
    #     assert output.shape == expected.shape, f"Shape mismatch: {output.shape} vs {expected.shape}"
    #     rtol = 1e-2 if is_fp16 else 1e-5
    #     atol = 1e-3 if is_fp16 else 1e-6

    #     assert torch.allclose(output, expected, rtol=rtol, atol=atol), (
    #         f"Results don't match. Max diff: {(output - expected).abs().max()}"
    #     )

    # def test_consistency_between_implementations(
    #     self, use_cuda, is_weights, scale_mode, is_fp16, input_size, block_size
    # ):
    #     """Test that all implementations produce consistent results"""
    #     if not torch.cuda.is_available() and use_cuda:
    #         pytest.skip("Skipping CUDA test cases for CPU only setups")

    #     skip_if_half_on_cpu(is_fp16, use_cuda)

    #     # Skip CPU tests since Triton requires CUDA
    #     if not use_cuda:
    #         pytest.skip("Triton requires CUDA")

    #     device = torch.device("cuda" if use_cuda else "cpu")
    #     dtype = torch.float16 if is_fp16 else torch.float32

    #     # Generate test data
    #     torch.manual_seed(42)
    #     input_tensor = torch.randn(input_size, device=device, dtype=dtype)
    #     ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

    #     # Compute results with all implementations
    #     pytorch_result = pytorch_sum_like(input_tensor, ref_tensor)
    #     triton_result = triton_sum_like(input_tensor, ref_tensor)

    #     # Test CPU sum_like for comparison (if input is small enough)
    #     if input_tensor.numel() < 10000:  # Only for small tensors to avoid memory issues
    #         cpu_input = input_tensor.cpu()
    #         cpu_ref = ref_tensor.cpu()
    #         cpu_result = sum_like(cpu_input, cpu_ref).to(device)

    #         rtol = 1e-2 if is_fp16 else 1e-5
    #         atol = 1e-3 if is_fp16 else 1e-6

    #         assert torch.allclose(cpu_result, pytorch_result, rtol=rtol, atol=atol), (
    #             f"CPU sum_like doesn't match PyTorch. Max diff: {(cpu_result - pytorch_result).abs().max()}"
    #         )

    #     # Test optimized kernel
    #     output_opt = torch.zeros_like(ref_tensor)
    #     input_meta = get_4d_tensor_meta(input_tensor)
    #     output_meta = get_4d_tensor_meta(output_opt)

    #     import triton

    #     grid_size = triton.cdiv(input_tensor.numel(), block_size)

    #     optimized_sum_reduction_kernel[(grid_size,)](
    #         input_tensor,
    #         input_meta,
    #         output_opt,
    #         output_meta,
    #         BLOCK_SIZE=block_size,
    #     )

    #     # Test working optimized kernel
    #     output_working = torch.zeros_like(ref_tensor)
    #     working_optimized_sum_reduction_kernel[(grid_size,)](
    #         input_tensor,
    #         input_meta,
    #         output_working,
    #         output_meta,
    #         BLOCK_SIZE=block_size,
    #     )

    #     # Check all results are consistent
    #     rtol = 1e-2 if is_fp16 else 1e-5
    #     atol = 1e-3 if is_fp16 else 1e-6

    #     assert torch.allclose(triton_result, pytorch_result, rtol=rtol, atol=atol), (
    #         f"triton_sum_like doesn't match PyTorch. Max diff: {(triton_result - pytorch_result).abs().max()}"
    #     )

    #     assert torch.allclose(output_opt, pytorch_result, rtol=rtol, atol=atol), (
    #         f"optimized_sum_reduction_kernel doesn't match PyTorch. Max diff: {(output_opt - pytorch_result).abs().max()}"
    #     )

    #     assert torch.allclose(output_working, pytorch_result, rtol=rtol, atol=atol), (
    #         f"working_optimized_sum_reduction_kernel doesn't match PyTorch. Max diff: {(output_working - pytorch_result).abs().max()}"
    #     )


# @pytest.mark.parametrize("use_cuda", [True], ids=["cuda"])  # Only CUDA since Triton requires it
# @pytest.mark.parametrize("is_weights", [True, False], ids=["weights", "activations"])
# @pytest.mark.parametrize("scale_mode", ["single_scale", "per_channel_scale"])
# @pytest.mark.parametrize("is_fp16", [True, False], ids=["fp16", "fp32"])
# @pytest.mark.parametrize("input_size", [[1, 16, 64, 64], [4, 16, 16, 16]], ids=idfn)
# class TestTritonSumReductionPerformance:
#     def test_block_size_performance(self, use_cuda, is_weights, scale_mode, is_fp16, input_size):
#         """Test performance with different block sizes"""
#         if not torch.cuda.is_available():
#             pytest.skip("Skipping CUDA test cases for CPU only setups")

#         skip_if_half_on_cpu(is_fp16, use_cuda)

#         device = torch.device("cuda")
#         dtype = torch.float16 if is_fp16 else torch.float32

#         # Generate test data
#         torch.manual_seed(42)
#         input_tensor = torch.randn(input_size, device=device, dtype=dtype)
#         ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

#         # Compute expected result
#         expected = pytorch_sum_like(input_tensor, ref_tensor)

#         # Test different block sizes
#         block_sizes = [256, 512, 1024]
#         results = {}

#         for block_size in block_sizes:
#             output = torch.zeros_like(ref_tensor)
#             input_meta = get_4d_tensor_meta(input_tensor)
#             output_meta = get_4d_tensor_meta(output)

#             import triton

#             grid_size = triton.cdiv(input_tensor.numel(), block_size)

#             # Warmup
#             for _ in range(3):
#                 optimized_sum_reduction_kernel[(grid_size,)](
#                     input_tensor,
#                     input_meta,
#                     output,
#                     output_meta,
#                     BLOCK_SIZE=block_size,
#                 )

#             # Time the kernel
#             torch.cuda.synchronize()
#             start = torch.cuda.Event(enable_timing=True)
#             end = torch.cuda.Event(enable_timing=True)

#             start.record()
#             for _ in range(10):
#                 optimized_sum_reduction_kernel[(grid_size,)](
#                     input_tensor,
#                     input_meta,
#                     output,
#                     output_meta,
#                     BLOCK_SIZE=block_size,
#                 )
#             end.record()
#             torch.cuda.synchronize()

#             elapsed_time = start.elapsed_time(end) / 10  # Average time per run
#             results[block_size] = {
#                 "time": elapsed_time,
#                 "grid_size": grid_size,
#                 "correct": torch.allclose(
#                     output, expected, rtol=1e-2 if is_fp16 else 1e-5, atol=1e-3 if is_fp16 else 1e-6
#                 ),
#             }

#         # Print results for debugging
#         print(f"\nPerformance results for {scale_mode}, {is_weights}, {is_fp16}, {input_size}:")
#         for block_size, result in results.items():
#             print(
#                 f"  Block size {block_size}: {result['time']:.3f}ms, Grid size: {result['grid_size']}, Correct: {result['correct']}"
#             )

#         # All results should be correct
#         for block_size, result in results.items():
#             assert result["correct"], f"Block size {block_size} produced incorrect results"


# @pytest.mark.parametrize(
#     "scale_mode", ["single_scale"]
# )  # Focus on single_scale since it has the most atomic contention
# @pytest.mark.parametrize("input_size", [[1, 16, 64, 64], [4, 16, 16, 16]], ids=idfn)
# class TestTritonSumReductionAtomicContention:
#     def test_atomic_contention_single_scale(self, scale_mode, input_size):
#         """Test atomic contention issues in single_scale mode"""
#         if not torch.cuda.is_available():
#             pytest.skip("Skipping CUDA test cases for CPU only setups")

#         device = torch.device("cuda")
#         dtype = torch.float16

#         # Generate test data
#         torch.manual_seed(42)
#         input_tensor = torch.randn(input_size, device=device, dtype=dtype)
#         ref_tensor = torch.ones([1], device=device, dtype=dtype)  # Single scale

#         # Compute expected result
#         expected = pytorch_sum_like(input_tensor, ref_tensor)

#         # Test with different block sizes to see atomic contention effects
#         block_sizes = [64, 128, 256, 512, 1024]
#         results = []

#         for block_size in block_sizes:
#             output = torch.zeros_like(ref_tensor)
#             input_meta = get_4d_tensor_meta(input_tensor)
#             output_meta = get_4d_tensor_meta(output)

#             import triton

#             grid_size = triton.cdiv(input_tensor.numel(), block_size)

#             # Run kernel multiple times to check consistency
#             outputs = []
#             for _ in range(5):
#                 current_output = torch.zeros_like(ref_tensor)
#                 optimized_sum_reduction_kernel[(grid_size,)](
#                     input_tensor,
#                     input_meta,
#                     current_output,
#                     output_meta,
#                     BLOCK_SIZE=block_size,
#                 )
#                 outputs.append(current_output.clone())

#             # Check consistency across runs
#             for i, out in enumerate(outputs):
#                 assert torch.allclose(out, outputs[0], rtol=1e-4, atol=1e-5), (
#                     f"Block size {block_size}, run {i}: Inconsistent results due to atomic contention"
#                 )

#             # Check correctness
#             correctness = torch.allclose(outputs[0], expected, rtol=1e-2, atol=1e-3)

#             results.append(
#                 {
#                     "block_size": block_size,
#                     "grid_size": grid_size,
#                     "correct": correctness,
#                     "result": outputs[0].item(),
#                     "expected": expected.item(),
#                 }
#             )

#         # Print results for debugging
#         print(f"\nAtomic contention test results for {input_size}:")
#         for result in results:
#             print(
#                 f"  Block size {result['block_size']}: Grid size {result['grid_size']}, "
#                 f"Correct: {result['correct']}, Result: {result['result']:.3f}, Expected: {result['expected']:.3f}"
#             )

#         # All results should be correct
#         for result in results:
#             assert result["correct"], (
#                 f"Block size {result['block_size']}: Result {result['result']:.3f} != Expected {result['expected']:.3f}"
#             )


# @pytest.mark.parametrize(
#     "kernel_name",
#     ["optimized_sum_reduction", "hierarchical_sum_like", "warp_level_sum_reduction", "warp_efficient_sum_like"],
# )
# @pytest.mark.parametrize("block_size", [64, 128, 256, 512, 1024])
# @pytest.mark.parametrize(
#     "input_size", [[1, 16, 64, 64], [4, 16, 16, 16], [8, 256, 32, 32], [1024, 256], [4096, 4096], [256]], ids=idfn
# )
# class TestTritonKernelBenchmark:
#     """Benchmark different Triton sum reduction kernels"""

#     def get_kernel_function(self, kernel_name: str):
#         """Get kernel function by name"""
#         from nncf.torch.quantization.triton.reference import hierarchical_sum_like_kernel
#         from nncf.torch.quantization.triton.reference import optimized_sum_reduction_kernel
#         from nncf.torch.quantization.triton.reference import warp_efficient_sum_like_kernel
#         from nncf.torch.quantization.triton.reference import warp_level_sum_reduction_kernel

#         kernel_map = {
#             "optimized_sum_reduction": optimized_sum_reduction_kernel,
#             "hierarchical_sum_like": hierarchical_sum_like_kernel,
#             "warp_level_sum_reduction": warp_level_sum_reduction_kernel,
#             "warp_efficient_sum_like": warp_efficient_sum_like_kernel,
#         }

#         if kernel_name not in kernel_map:
#             pytest.skip(f"Kernel {kernel_name} not available")

#         return kernel_map[kernel_name]

#     def benchmark_kernel_performance(self, kernel_func, input_tensor, ref_tensor, block_size, runs=50):
#         """Benchmark kernel performance"""
#         import triton

#         # Prepare kernel inputs
#         output = torch.zeros_like(ref_tensor)
#         input_meta = get_4d_tensor_meta(input_tensor)
#         output_meta = get_4d_tensor_meta(output)
#         grid_size = triton.cdiv(input_tensor.numel(), block_size)

#         # Warmup
#         for _ in range(10):
#             try:
#                 kernel_func[(grid_size,)](input_tensor, input_meta, output, output_meta, BLOCK_SIZE=block_size)
#             except Exception as e:
#                 pytest.skip(f"Kernel failed: {e}")

#         # Benchmark
#         torch.cuda.synchronize()
#         start_time = torch.cuda.Event(enable_timing=True)
#         end_time = torch.cuda.Event(enable_timing=True)

#         start_time.record()
#         for _ in range(runs):
#             kernel_func[(grid_size,)](input_tensor, input_meta, output, output_meta, BLOCK_SIZE=block_size)
#         end_time.record()

#         torch.cuda.synchronize()
#         elapsed_time = start_time.elapsed_time(end_time)
#         avg_time = elapsed_time / runs

#         return avg_time, output

#     def test_kernel_performance_benchmark(self, kernel_name, block_size, input_size, is_weights=False):
#         """Test and benchmark kernel performance"""
#         if not torch.cuda.is_available():
#             pytest.skip("CUDA not available")

#         scale_mode = "per_channel_scale"
#         device = torch.device("cuda")
#         dtype = torch.float16

#         # Generate test data
#         torch.manual_seed(42)
#         input_tensor = torch.randn(input_size, device=device, dtype=dtype)
#         ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

#         # Get kernel function
#         kernel_func = self.get_kernel_function(kernel_name)

#         # Benchmark performance
#         avg_time, output = self.benchmark_kernel_performance(kernel_func, input_tensor, ref_tensor, block_size)

#         # Verify correctness
#         expected = pytorch_sum_like(input_tensor, ref_tensor)
#         rtol = 1e-2
#         atol = 1e-3

#         assert torch.allclose(output, expected, rtol=rtol, atol=atol), (
#             f"Kernel {kernel_name} produced incorrect results. Max diff: {(output - expected).abs().max()}"
#         )

#         # Calculate throughput
#         input_bytes = input_tensor.numel() * input_tensor.element_size()
#         output_bytes = output.numel() * output.element_size()
#         total_bytes = input_bytes + output_bytes
#         throughput_gb_s = (total_bytes / 1e9) / (avg_time / 1000)

#         # Print results for analysis
#         print(
#             f"\n{kernel_name:25s} | {str(input_size):20s} | bs={block_size:4d} | "
#             f"{avg_time:7.3f}ms | {throughput_gb_s:6.1f}GB/s"
#         )

#         # Store results for later analysis (could be extended to write to file)
#         result = {
#             "kernel": kernel_name,
#             "input_size": input_size,
#             "block_size": block_size,
#             "is_weights": is_weights,
#             "avg_time_ms": avg_time,
#             "throughput_gb_s": throughput_gb_s,
#             "input_numel": input_tensor.numel(),
#             "ref_numel": ref_tensor.numel(),
#         }

#         return result


# @pytest.mark.parametrize("use_optimal_block_size", [True, False])
# @pytest.mark.parametrize(
#     "input_size", [[1, 16, 64, 64], [4, 16, 16, 16], [8, 256, 32, 32], [1024, 256], [4096, 4096]], ids=idfn
# )
# class TestOptimalBlockSizeEffectiveness:
#     """Test the effectiveness of optimal block size selection"""

#     def test_optimal_vs_fixed_block_size(self, use_optimal_block_size, input_size, is_weights=False):
#         """Compare optimal vs fixed block size performance"""
#         if not torch.cuda.is_available():
#             pytest.skip("CUDA not available")

#         scale_mode = "per_channel_scale"
#         device = torch.device("cuda")
#         dtype = torch.float16

#         # Generate test data
#         torch.manual_seed(42)
#         input_tensor = torch.randn(input_size, device=device, dtype=dtype)
#         ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

#         # Calculate optimal block size
#         from nncf.torch.quantization.triton.reference import calculate_contiguous_elements_per_scale
#         from nncf.torch.quantization.triton.reference import optimize_block_size_for_contiguous_elements

#         contiguous_elements = calculate_contiguous_elements_per_scale(input_tensor, ref_tensor)
#         optimal_block_size = optimize_block_size_for_contiguous_elements(contiguous_elements, input_tensor.numel())

#         # Choose block size based on test parameter
#         if use_optimal_block_size:
#             block_size = optimal_block_size
#             block_type = "optimal"
#         else:
#             block_size = 256  # Fixed default
#             block_type = "fixed"

#         # Benchmark with triton_sum_like
#         runs = 100
#         torch.cuda.synchronize()
#         start_time = torch.cuda.Event(enable_timing=True)
#         end_time = torch.cuda.Event(enable_timing=True)

#         # Warmup
#         for _ in range(10):
#             if use_optimal_block_size:
#                 result = triton_sum_like(input_tensor, ref_tensor)  # Uses automatic optimization
#             else:
#                 result = triton_sum_like(input_tensor, ref_tensor, block_size=block_size)

#         # Benchmark
#         start_time.record()
#         for _ in range(runs):
#             if use_optimal_block_size:
#                 result = triton_sum_like(input_tensor, ref_tensor)
#             else:
#                 result = triton_sum_like(input_tensor, ref_tensor, block_size=block_size)
#         end_time.record()

#         torch.cuda.synchronize()
#         elapsed_time = start_time.elapsed_time(end_time)
#         avg_time = elapsed_time / runs

#         # Verify correctness
#         expected = pytorch_sum_like(input_tensor, ref_tensor)
#         rtol = 1e-2
#         atol = 1e-3

#         assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
#             f"triton_sum_like with {block_type} block size produced incorrect results"
#         )

#         # Calculate throughput
#         input_bytes = input_tensor.numel() * input_tensor.element_size()
#         output_bytes = result.numel() * result.element_size()
#         total_bytes = input_bytes + output_bytes
#         throughput_gb_s = (total_bytes / 1e9) / (avg_time / 1000)

#         # Print results
#         print(
#             f"\n{block_type:8s} block size | {str(input_size):20s} | bs={block_size:4d} | "
#             f"contiguous={contiguous_elements:8d} | {avg_time:7.3f}ms | {throughput_gb_s:6.1f}GB/s"
#         )

#         # Store results for comparison
#         return {
#             "block_type": block_type,
#             "block_size": block_size,
#             "optimal_block_size": optimal_block_size,
#             "input_size": input_size,
#             "contiguous_elements": contiguous_elements,
#             "avg_time_ms": avg_time,
#             "throughput_gb_s": throughput_gb_s,
#             "is_optimal": use_optimal_block_size,
#         }


# @pytest.mark.parametrize("use_cuda", [True], ids=["cuda"])  # Only CUDA since Triton requires it
# @pytest.mark.parametrize("is_weights", [True, False], ids=["weights", "activations"])
# @pytest.mark.parametrize("scale_mode", ["single_scale", "per_channel_scale"])
# @pytest.mark.parametrize("is_fp16", [True, False], ids=["fp16", "fp32"])
# @pytest.mark.parametrize("input_size", [[1, 16, 64, 64], [4, 16, 16, 16]], ids=idfn)
# class TestTritonSumReductionPerformance:
#     def test_block_size_performance(self, use_cuda, is_weights, scale_mode, is_fp16, input_size):
#         """Test performance with different block sizes"""
#         if not torch.cuda.is_available():
#             pytest.skip("Skipping CUDA test cases for CPU only setups")

#         skip_if_half_on_cpu(is_fp16, use_cuda)

#         device = torch.device("cuda")
#         dtype = torch.float16 if is_fp16 else torch.float32

#         # Generate test data
#         torch.manual_seed(42)
#         input_tensor = torch.randn(input_size, device=device, dtype=dtype)
#         ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

#         # Compute expected result
#         expected = pytorch_sum_like(input_tensor, ref_tensor)

#         # Test different block sizes
#         block_sizes = [256, 512, 1024]
#         results = {}

#         for block_size in block_sizes:
#             output = torch.zeros_like(ref_tensor)
#             input_meta = get_4d_tensor_meta(input_tensor)
#             output_meta = get_4d_tensor_meta(output)

#             import triton

#             grid_size = triton.cdiv(input_tensor.numel(), block_size)

#             # Warmup
#             for _ in range(3):
#                 optimized_sum_reduction_kernel[(grid_size,)](
#                     input_tensor,
#                     input_meta,
#                     output,
#                     output_meta,
#                     BLOCK_SIZE=block_size,
#                 )

#             # Time the kernel
#             torch.cuda.synchronize()
#             start = torch.cuda.Event(enable_timing=True)
#             end = torch.cuda.Event(enable_timing=True)

#             start.record()
#             for _ in range(10):
#                 optimized_sum_reduction_kernel[(grid_size,)](
#                     input_tensor,
#                     input_meta,
#                     output,
#                     output_meta,
#                     BLOCK_SIZE=block_size,
#                 )
#             end.record()
#             torch.cuda.synchronize()

#             elapsed_time = start.elapsed_time(end) / 10  # Average time per run
#             results[block_size] = {
#                 "time": elapsed_time,
#                 "grid_size": grid_size,
#                 "correct": torch.allclose(
#                     output, expected, rtol=1e-2 if is_fp16 else 1e-5, atol=1e-3 if is_fp16 else 1e-6
#                 ),
#             }

#         # Print results for debugging
#         print(f"\nPerformance results for {scale_mode}, {is_weights}, {is_fp16}, {input_size}:")
#         for block_size, result in results.items():
#             print(
#                 f"  Block size {block_size}: {result['time']:.3f}ms, Grid size: {result['grid_size']}, Correct: {result['correct']}"
#             )

#         # All results should be correct
#         for block_size, result in results.items():
#             assert result["correct"], f"Block size {block_size} produced incorrect results"


# @pytest.mark.parametrize(
#     "scale_mode", ["single_scale"]
# )  # Focus on single_scale since it has the most atomic contention
# @pytest.mark.parametrize("input_size", [[1, 16, 64, 64], [4, 16, 16, 16]], ids=idfn)
# class TestTritonSumReductionAtomicContention:
#     def test_atomic_contention_single_scale(self, scale_mode, input_size):
#         """Test atomic contention issues in single_scale mode"""
#         if not torch.cuda.is_available():
#             pytest.skip("Skipping CUDA test cases for CPU only setups")

#         device = torch.device("cuda")
#         dtype = torch.float16

#         # Generate test data
#         torch.manual_seed(42)
#         input_tensor = torch.randn(input_size, device=device, dtype=dtype)
#         ref_tensor = torch.ones([1], device=device, dtype=dtype)  # Single scale

#         # Compute expected result
#         expected = pytorch_sum_like(input_tensor, ref_tensor)

#         # Test with different block sizes to see atomic contention effects
#         block_sizes = [64, 128, 256, 512, 1024]
#         results = []

#         for block_size in block_sizes:
#             output = torch.zeros_like(ref_tensor)
#             input_meta = get_4d_tensor_meta(input_tensor)
#             output_meta = get_4d_tensor_meta(output)

#             import triton

#             grid_size = triton.cdiv(input_tensor.numel(), block_size)

#             # Run kernel multiple times to check consistency
#             outputs = []
#             for _ in range(5):
#                 current_output = torch.zeros_like(ref_tensor)
#                 optimized_sum_reduction_kernel[(grid_size,)](
#                     input_tensor,
#                     input_meta,
#                     current_output,
#                     output_meta,
#                     BLOCK_SIZE=block_size,
#                 )
#                 outputs.append(current_output.clone())

#             # Check consistency across runs
#             for i, out in enumerate(outputs):
#                 assert torch.allclose(out, outputs[0], rtol=1e-4, atol=1e-5), (
#                     f"Block size {block_size}, run {i}: Inconsistent results due to atomic contention"
#                 )

#             # Check correctness
#             correctness = torch.allclose(outputs[0], expected, rtol=1e-2, atol=1e-3)

#             results.append(
#                 {
#                     "block_size": block_size,
#                     "grid_size": grid_size,
#                     "correct": correctness,
#                     "result": outputs[0].item(),
#                     "expected": expected.item(),
#                 }
#             )

#         # Print results for debugging
#         print(f"\nAtomic contention test results for {input_size}:")
#         for result in results:
#             print(
#                 f"  Block size {result['block_size']}: Grid size {result['grid_size']}, "
#                 f"Correct: {result['correct']}, Result: {result['result']:.3f}, Expected: {result['expected']:.3f}"
#             )

#         # All results should be correct
#         for result in results:
#             assert result["correct"], (
#                 f"Block size {result['block_size']}: Result {result['result']:.3f} != Expected {result['expected']:.3f}"
#             )


if __name__ == "__main__":
    # Run a simple test
    test = TestTritonSumReduction()
    test.test_triton_sum_like_correctness(
        use_cuda=True,
        is_weights=False,
        scale_mode="single_scale",
        is_fp16=False,
        input_size=[4, 16, 16, 16],
        block_size=256,
    )
    print("Simple test passed!")
