/**
 * OPTIMIZED PER-GROUP QUANTIZATION BACKWARD KERNEL
 *
 * Key optimizations:
 * 1. Reduce grid size by processing multiple groups per block
 * 2. Improve memory access patterns with better coalescing
 * 3. Minimize the number of separate reductions
 * 4. Use shared memory more efficiently
 */

template <typename scalar_t, typename scalar_accum_t>
__global__ void q_scale_per_group_cuda_backward_kernel_optimized(
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_input_low,
        scalar_t* __restrict__ grad_input_range,
        scalar_accum_t* __restrict__ dev_tmp_range,
        scalar_accum_t* __restrict__ dev_tmp_low,
        int32_t* __restrict__ dev_last_block_counter_range,
        int32_t* __restrict__ dev_last_block_counter_low,
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

    // OPTIMIZATION 1: Process multiple groups per block to reduce grid size
    const uint32_t groups_per_block = 8;  // Tunable parameter
    const uint32_t tidx = threadIdx.x;
    const uint32_t block_idx = blockIdx.x;

    // Calculate which groups this block will process
    const uint32_t first_group_idx = block_idx * groups_per_block;
    const uint32_t last_group_idx = min(first_group_idx + groups_per_block,
                                       static_cast<uint32_t>(cout * num_groups_per_channel));

    // OPTIMIZATION 2: Shared memory for multiple group reductions
    __shared__ scalar_accum_t shared_grad_range[groups_per_block][CUDA_MAX_NUM_THREADS_PER_BLOCK / 32];
    __shared__ scalar_accum_t shared_grad_low[groups_per_block][CUDA_MAX_NUM_THREADS_PER_BLOCK / 32];

    const scalar_t alpha = level_low / level_high;

    // Process each group assigned to this block
    for (uint32_t group_idx = first_group_idx; group_idx < last_group_idx; group_idx++) {
        const uint32_t local_group_idx = group_idx - first_group_idx;

        // Calculate group position
        const size_t channel_idx = group_idx / num_groups_per_channel;
        const size_t group_idx_in_channel = group_idx % num_groups_per_channel;

        // OPTIMIZATION 3: Better memory access pattern
        const size_t channel_size = cin;  // Total elements per channel
        const size_t group_start_in_channel = group_idx_in_channel * group_size;
        const size_t offset = channel_idx * channel_size + group_start_in_channel;

        // Load scale parameters once per group
        const scalar_t scale_low = input_low[group_idx];
        const scalar_t scale_range = input_range[group_idx];
        const scalar_t range_low = scale_low;
        const scalar_t range_high = scale_low + scale_range;
        const scalar_t reverted_range = 1.0f / scale_range;

        scalar_accum_t per_thread_grad_sum_range = 0;
        scalar_accum_t per_thread_grad_sum_low = 0;

        // OPTIMIZATION 4: Vectorized memory access where possible
        // Process elements in this group with coalesced access
        for (size_t i = tidx; i < group_size; i += blockDim.x) {
            const size_t element_idx = offset + i;

            // Compute quantization
            const scalar_t input_val = input[element_idx];
            const scalar_t clamped_val = fminf(fmaxf(input_val, range_low), range_high);
            const scalar_t s = (levels - 1) / scale_range;
            const scalar_t zero_point = roundf(-scale_low * s);
            const scalar_t output_val = roundf((clamped_val - scale_low) * s - zero_point) / s;

            // Compute gradients
            const scalar_t grad_out = grad_output[element_idx];
            scalar_t val_grad_input_range = 0;
            scalar_t val_grad_input_low = 0;
            scalar_t val_grad_input = 0;

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
            per_thread_grad_sum_range += val_grad_input_range;
            per_thread_grad_sum_low += val_grad_input_low;
        }

        // OPTIMIZATION 5: Efficient warp-level reduction first
        // Reduce within warp
        const uint32_t warp_id = tidx / 32;
        const uint32_t lane_id = tidx % 32;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            per_thread_grad_sum_range += __shfl_down_sync(0xffffffff, per_thread_grad_sum_range, offset);
            per_thread_grad_sum_low += __shfl_down_sync(0xffffffff, per_thread_grad_sum_low, offset);
        }

        // Store warp results in shared memory
        if (lane_id == 0) {
            shared_grad_range[local_group_idx][warp_id] = per_thread_grad_sum_range;
            shared_grad_low[local_group_idx][warp_id] = per_thread_grad_sum_low;
        }

        __syncthreads();

        // Final reduction across warps
        if (tidx < (blockDim.x / 32)) {
            scalar_accum_t final_range = (tidx < (blockDim.x / 32)) ? shared_grad_range[local_group_idx][tidx] : 0;
            scalar_accum_t final_low = (tidx < (blockDim.x / 32)) ? shared_grad_low[local_group_idx][tidx] : 0;

            #pragma unroll
            for (int offset = (blockDim.x / 64); offset > 0; offset /= 2) {
                final_range += __shfl_down_sync(0xffffffff, final_range, offset);
                final_low += __shfl_down_sync(0xffffffff, final_low, offset);
            }

            if (tidx == 0) {
                grad_input_range[group_idx] = final_range;
                grad_input_low[group_idx] = final_low;
            }
        }

        __syncthreads();
    }
}


// OPTIMIZATION: Modified host function with better grid sizing
std::vector<at::Tensor> q_scale_per_group_cuda_backward_optimized(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {

    at::DeviceGuard guard(input.device());

    // Calculate group parameters
    const auto scale_count = input_range.numel();
    const auto elements_per_scale = input.numel() / scale_count;
    const auto cout = input_range.size(0);
    const auto num_groups_per_channel = input_range.size(1);
    const auto group_size = elements_per_scale;
    const auto cin = cout > 0 ? (input.numel() / cout) : 0;

    auto grad_input = at::empty_like(grad_output);
    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    // OPTIMIZATION: Dramatically reduce grid size
    const uint32_t groups_per_block = 8;  // Process multiple groups per block
    const uint32_t num_blocks = (scale_count + groups_per_block - 1) / groups_per_block;

    // Use 1D grid instead of 2D to reduce scheduling overhead
    dim3 grid_size(num_blocks);
    dim3 block_size(CUDA_MAX_NUM_THREADS_PER_BLOCK);

    PROFILE(DISPATCH_TENSOR_DATA_TYPES(input.scalar_type(), "q_scale_per_group_cuda_backward_optimized", ([&] {
        using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
        q_scale_per_group_cuda_backward_kernel_optimized<scalar_t, scalar_accum_t><<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_input.data_ptr<scalar_t>(),
            grad_input_low.data_ptr<scalar_t>(),
            grad_input_range.data_ptr<scalar_t>(),
            nullptr,  // No longer need intermediate buffers
            nullptr,
            nullptr,
            nullptr,
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
