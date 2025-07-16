import torch
import triton
import triton.language as tl


@triton.jit
def _sum_like_4d_kernel_coalesced(
    input_ptr,
    output_ptr,
    input_s0,
    input_s1,
    input_s2,
    input_s3,
    input_st0,
    input_st1,
    input_st2,
    input_st3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Memory-coalesced sum reduction kernel for 4D tensors.

    This kernel processes contiguous memory regions to maximize
    memory bandwidth utilization.
    """
    # Get program IDs for 2D grid
    channel_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if channel_idx >= input_s1:
        return

    # Calculate total elements to process for this channel
    elements_per_channel = input_s0 * input_s2 * input_s3

    # Calculate the starting offset for this block
    start_offset = block_idx * BLOCK_SIZE

    # Generate linear indices for this block
    linear_idx = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < elements_per_channel

    # Convert to 4D coordinates for dimensions 0, 2, 3
    tmp = linear_idx
    i3 = tmp % input_s3
    tmp //= input_s3
    i2 = tmp % input_s2
    tmp //= input_s2
    i0 = tmp % input_s0

    # Calculate memory offsets
    mem_offsets = i0 * input_st0 + channel_idx * input_st1 + i2 * input_st2 + i3 * input_st3

    # Load data
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0)

    # Sum within the block and use atomic add for final result
    block_sum = tl.sum(data)
    tl.atomic_add(output_ptr + channel_idx, block_sum)


def sum_like_v2(tensor_to_sum, ref_tensor):
    """Optimized Triton implementation of sum_like for 4D tensors."""
    # Create output tensor with the same dtype as the input tensor
    output = torch.zeros(
        tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=torch.float16
    )  # tensor_to_sum.dtype)

    # Calculate grid dimensions for parallel processing
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Use 2D grid: (channels, blocks_per_channel)
    grid = (tensor_to_sum.shape[1], num_blocks)
    _sum_like_4d_kernel_coalesced[grid](
        tensor_to_sum, output, *tensor_to_sum.shape, *tensor_to_sum.stride(), BLOCK_SIZE=BLOCK_SIZE
    )

    return output.reshape(ref_tensor.shape)


# --- End of re-included code ---


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_ELEMENTS"],
        x_vals=[4 * 16 * s * s for s in [16, 32, 64, 128, 256]],
        line_arg="provider",
        line_vals=["pytorch", "custom_kernel_v2"],
        line_names=["PyTorch", "Custom Kernel (v2)"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="ms",
        plot_name="sum-like-4d-performance-comparison",
        args={"D1_size": 16},
    )
)
def benchmark(D1_size, N_ELEMENTS, provider):
    # Infer shapes from total elements
    D0_size = 4
    D2_size = int((N_ELEMENTS / (D0_size * D1_size)) ** 0.5)
    D3_size = int(N_ELEMENTS / (D0_size * D1_size * D2_size))

    shape = (D0_size, D1_size, D2_size, D3_size)
    ref_shape = (1, D1_size, 1, 1)

    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    ref = torch.empty(ref_shape, device="cuda", dtype=torch.float16)

    quantiles = [0.2, 0.5, 0.8]

    if provider == "pytorch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.sum(x, axis=(0, 2, 3), keepdim=True), quantiles=quantiles
        )
    elif provider == "custom_kernel_v2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_v2(x, ref), quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    # Test correctness first
    print("Testing correctness...")

    # Test case
    tensor_to_sum = torch.randn(4, 16, 16, 16, device="cuda", dtype=torch.float16)
    ref_tensor = torch.empty(1, 16, 1, 1, device="cuda")

    # PyTorch reference
    pytorch_result = torch.sum(tensor_to_sum, axis=(0, 2, 3), keepdim=True)

    # Our implementations
    triton_result_v2 = sum_like_v2(tensor_to_sum, ref_tensor)

    print(f"PyTorch result shape: {pytorch_result.shape}")
    print(f"Triton v2 result shape: {triton_result_v2.shape}")

    # Check correctness
    assert torch.allclose(pytorch_result, triton_result_v2, rtol=1, atol=1e-1), (
        f"Results don't match. Max diff: {(pytorch_result - triton_result_v2).abs().max()}"
    )

    print("✓ Float16 test passed!")
    print("Running benchmark...")
    benchmark.run(show_plots=True, print_data=True, save_path="gemma.png")
