/**
 * PRACTICAL OPTIMIZATION RECOMMENDATIONS FOR q_scale_per_group_cuda_backward_kernel
 *
 * Based on analysis of the 125ms performance issue with [2048, 128256] input shape.
 */

// PROBLEM ANALYSIS:
// Current: 2,052,096 blocks, each processing only 128 elements
// This creates massive GPU scheduling overhead and poor resource utilization

// SOLUTION 1: REDUCE GRID SIZE - Process multiple groups per block
// Instead of 1 group per block, process N groups per block to reduce scheduling overhead

template <typename scalar_t, typename scalar_accum_t>
__global__ void q_scale_per_group_cuda_backward_kernel_v2(
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_input_low,
        scalar_t* __restrict__ grad_input_range,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const scalar_t level_low,
        const scalar_t level_high,
        const size_t group_size,
        const size_t num_groups_per_channel,
        const size_t cout,
        const size_t cin) {

    // KEY OPTIMIZATION: Process multiple groups per block
    const int GROUPS_PER_BLOCK = 4;  // Tunable: start with 4, test 8, 16

    const int tidx = threadIdx.x;
    const int block_id = blockIdx.x;

    // Calculate which groups this block handles
    const int first_group = block_id * GROUPS_PER_BLOCK;
    const int last_group = min(first_group + GROUPS_PER_BLOCK,
                              static_cast<int>(cout * num_groups_per_channel));

    const scalar_t alpha = level_low / level_high;

    // Process each group sequentially within the block
    for (int group_idx = first_group; group_idx < last_group; group_idx++) {
        // Map group_idx to [channel, group_in_channel]
        const int channel_idx = group_idx / num_groups_per_channel;
        const int group_in_channel = group_idx % num_groups_per_channel;

        // Calculate memory offset for this group
        const size_t group_offset = channel_idx * cin + group_in_channel * group_size;

        // Load scale parameters for this group
        const scalar_t scale_low = input_low[group_idx];
        const scalar_t scale_range = input_range[group_idx];
        const scalar_t range_low = scale_low;
        const scalar_t range_high = scale_low + scale_range;
        const scalar_t reverted_range = 1.0f / scale_range;

        // Accumulate gradients for this group
        scalar_accum_t sum_range = 0;
        scalar_accum_t sum_low = 0;

        // Each thread processes elements within the group
        for (int i = tidx; i < group_size; i += blockDim.x) {
            const size_t element_idx = group_offset + i;

            // Compute fake quantization (inlined for performance)
            const scalar_t input_val = input[element_idx];
            const scalar_t s = (levels - 1) / scale_range;
            const scalar_t zero_point = roundf(-scale_low * s);
            const scalar_t clamped = fminf(fmaxf(input_val, range_low), range_high);
            const scalar_t output_val = roundf((clamped - scale_low) * s - zero_point) / s;

            // Compute gradients (inlined for performance)
            const scalar_t grad_out = grad_output[element_idx];
            scalar_t val_grad_input = 0;
            scalar_t val_grad_input_range = 0;
            scalar_t val_grad_input_low = 0;

            if (input_val < range_low) {
                val_grad_input_range = alpha * grad_out;
                val_grad_input_low = grad_out;
            } else if (input_val > range_high) {
                val_grad_input_range = grad_out;
                val_grad_input_low = grad_out;
            } else {
                val_grad_input_range = grad_out * ((output_val - input_val) * reverted_range);
                val_grad_input = grad_out;
            }

            grad_input[element_idx] = val_grad_input;
            sum_range += val_grad_input_range;
            sum_low += val_grad_input_low;
        }

        // OPTIMIZED REDUCTION: Use warp shuffles first, then shared memory
        __shared__ scalar_accum_t sdata_range[32];
        __shared__ scalar_accum_t sdata_low[32];

        const int warp_id = tidx / 32;
        const int lane_id = tidx % 32;

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_range += __shfl_down_sync(0xffffffff, sum_range, offset);
            sum_low += __shfl_down_sync(0xffffffff, sum_low, offset);
        }

        // Store warp results
        if (lane_id == 0) {
            sdata_range[warp_id] = sum_range;
            sdata_low[warp_id] = sum_low;
        }

        __syncthreads();

        // Final reduction across warps
        if (tidx < 32) {
            scalar_accum_t val_range = (tidx < blockDim.x / 32) ? sdata_range[tidx] : 0;
            scalar_accum_t val_low = (tidx < blockDim.x / 32) ? sdata_low[tidx] : 0;

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val_range += __shfl_down_sync(0xffffffff, val_range, offset);
                val_low += __shfl_down_sync(0xffffffff, val_low, offset);
            }

            if (tidx == 0) {
                grad_input_range[group_idx] = val_range;
                grad_input_low[group_idx] = val_low;
            }
        }

        __syncthreads();  // Ensure all threads are ready for next group
    }
}

// OPTIMIZED HOST FUNCTION
std::vector<at::Tensor> q_scale_per_group_cuda_backward_v2(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {

    at::DeviceGuard guard(input.device());

    const auto scale_count = input_range.numel();
    const auto elements_per_scale = input.numel() / scale_count;
    const auto cout = input_range.size(0);
    const auto num_groups_per_channel = input_range.size(1);
    const auto group_size = elements_per_scale;
    const auto cin = input.numel() / cout;

    auto grad_input = at::empty_like(grad_output);
    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    // CRITICAL OPTIMIZATION: Dramatically reduce grid size
    const int GROUPS_PER_BLOCK = 4;
    const int num_blocks = (scale_count + GROUPS_PER_BLOCK - 1) / GROUPS_PER_BLOCK;

    // Use 1D grid - much simpler and faster than 2D
    dim3 grid_size(num_blocks);
    dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK);

    PROFILE(DISPATCH_TENSOR_DATA_TYPES(input.scalar_type(), "q_scale_per_group_cuda_backward_v2", ([&] {
        using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
        q_scale_per_group_cuda_backward_kernel_v2<scalar_t, scalar_accum_t><<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_input.data_ptr<scalar_t>(),
            grad_input_low.data_ptr<scalar_t>(),
            grad_input_range.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input_low.data_ptr<scalar_t>(),
            input_range.data_ptr<scalar_t>(),
            levels,
            level_low,
            level_high,
            group_size,
            num_groups_per_channel,
            cout,
            cin);
    }));

    return {grad_input, grad_input_low, grad_input_range};
}

/*
EXPECTED PERFORMANCE IMPROVEMENT:

Current: 2,052,096 blocks
Optimized: 513,024 blocks (4x reduction)

Benefits:
1. 4x fewer block launches = ~75% reduction in scheduling overhead
2. Better GPU utilization (each block does more work)
3. Reduced memory allocation overhead (no intermediate buffers)
4. Faster warp-level reductions
5. Simplified 1D grid vs 2D grid

Expected speedup: 3-5x (from 125ms to 25-40ms)

Further optimizations to try:
- Increase GROUPS_PER_BLOCK to 8 or 16
- Use shared memory for input data caching
- Vectorized memory access (float4, etc.)
- Triton implementation for automatic optimization
*/
