#include "common_cuda_funcs.cuh"
#include "common_cuda_defs.cuh"


enum class ScaleType
{
    SINGLE_SCALE,
    PER_WEIGHT_CHANNEL,
    PER_ACTIVATION_CHANNEL,
    PER_GROUP,
};


ScaleType get_scale_type(const at::Tensor& input, const at::Tensor& input_low, const at::Tensor& input_range)
{
    TORCH_CHECK(input_low.dim() == input_range.dim(), "input_low and input_range have different dimensionality");
    uint64_t scale_dim = input_range.dim();
    for (int i = 0; i < scale_dim; i++)
    {
        TORCH_CHECK(input_low.size(i) == input_range.size(i), "input_low and input_range have different dimension sizes");
    }

    uint64_t scale_count = input_range.numel();

    if (scale_dim > 0)
    {
        // For per-group quantization, scale tensor has shape [Cout, Cin//GS, 1] (3D)
        // For per-channel quantization, scale tensor has shape [Cout, 1] or [1, Cin] (2D or effectively 1D)
        if (scale_dim >= 3 || (scale_dim == 2 && input_range.size(0) > 1 && input_range.size(1) > 1))
        {
            // This indicates per-group quantization: scale shape is [Cout, Cin//GS, 1] or similar multi-dimensional
            return ScaleType::PER_GROUP;
        }

        // For (NxCxHxW) input/output tensors, it is assumed that input_range is
        // either (1) for single-scale quantization, or (Nx1x1x1) for
        // per-channel scale weights quantization, or (1xCx1x1) for per-channel
        // activation quantization
        if (input_range.size(0) > 1)
        {
            TORCH_CHECK(input_range.size(0) == input.size(0), "Scale count and weights input channel count is different");
            TORCH_CHECK(input_range.size(0) == scale_count, "Scale shape is not flat");
            return ScaleType::PER_WEIGHT_CHANNEL;
        }
        else if (scale_dim >= 2 && input_range.size(1) > 1)
        {
            TORCH_CHECK(input_range.size(1) == input.size(1), "Scale count and activations channel count is different");
            TORCH_CHECK(input_range.size(1) == scale_count, "Scale shape is not flat");
            return  ScaleType::PER_ACTIVATION_CHANNEL;
        }
        // For (1x1x1x1) input/output tensors, it is assumed that input_range
        // should be PER_WEIGHT_CHANNEL
        if (scale_count == 1)
            return ScaleType::PER_WEIGHT_CHANNEL;
    }

    return ScaleType::SINGLE_SCALE;
}


namespace {

template <typename scalar_t>
__device__ void fakeQuantize(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels
        ) {
    scalar_t s = (levels - 1) / (*input_range);
    // zero_point is referred as ZP in docs
    scalar_t zero_point = round((-(*input_low) * s));
    (*output) = round((min(max((*input), (*input_low)), (*input_low) + (*input_range)) - (*input_low)) * s - zero_point) / s;
}

template <typename scalar_t>
__global__ void q_cuda_forward_kernel(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const uint64_t size,
        const uint64_t contiguous_elements_per_scale,
        const uint64_t scale_count) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // "Scales" are derived from input_low/input_range
        uint64_t scale_idx = static_cast<uint64_t>(idx / contiguous_elements_per_scale) % scale_count;
        fakeQuantize<scalar_t>((output + idx), (input + idx), input_low + scale_idx, input_range + scale_idx, levels);
    }
}

template <typename scalar_t>
__device__ void calcGrad(
        scalar_t* __restrict__ val_grad_input,
        scalar_t* __restrict__ val_grad_input_low,
        scalar_t* __restrict__ val_grad_input_range,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ output,
        const scalar_t range_low,
        const scalar_t range_high,
        const scalar_t reverted_range,
        const scalar_t val_low_grad) {
    *val_grad_input_range = 0;
    *val_grad_input_low = 0;
    *val_grad_input = 0;

    if ((*input) < range_low) {
        (*val_grad_input_range) = val_low_grad * (*grad_output);
        (*val_grad_input_low) = (*grad_output);
    } else if ((*input) > range_high) {
        (*val_grad_input_range) = (*grad_output);
        (*val_grad_input_low) = (*grad_output);
    } else {
        (*val_grad_input_range) = (*grad_output) * (((*output) - (*input)) * reverted_range);
        (*val_grad_input) = (*grad_output);
    }
}


template <typename scalar_t, typename scalar_accum_t>
__global__ void q_single_scale_cuda_backward_kernel(
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
        const size_t size) {
    const uint16_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint64_t gtidx = bidx * CUDA_MAX_NUM_THREADS_PER_BLOCK + tidx;
    const uint64_t grid_size = CUDA_MAX_NUM_THREADS_PER_BLOCK * gridDim.x;

    scalar_accum_t sum_range = 0, sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low);
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);
    for (size_t i = gtidx; i < size; i += grid_size) {
        fakeQuantize<scalar_t>(&output, (input + i), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + i), &val_grad_input_low, &val_grad_input_range, (grad_output + i),
                 (input + i), &output, range_low, range_high, reverted_range, alpha);
        sum_range += val_grad_input_range;
        sum_low += val_grad_input_low;
    }

    __shared__ scalar_accum_t sh_grad_range[CUDA_MAX_NUM_THREADS_PER_BLOCK];
    __shared__ scalar_accum_t sh_grad_low[CUDA_MAX_NUM_THREADS_PER_BLOCK];
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(sh_grad_range, sum_range, tidx, bidx, dev_tmp_range, dev_last_block_counter_range, grad_input_range, gridDim.x);
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(sh_grad_low, sum_low, tidx, bidx, dev_tmp_low, dev_last_block_counter_low, grad_input_low, gridDim.x);
}



template <typename scalar_t, typename scalar_accum_t>
__global__ void q_scale_per_weight_channel_cuda_backward_kernel(
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
        const size_t elements_per_scale) {
    const uint16_t tidx = threadIdx.x;
    const uint32_t scale_idx = blockIdx.x;
    const uint32_t per_scale_block_idx = blockIdx.y;

    const uint64_t per_scale_tidx = per_scale_block_idx * CUDA_MAX_NUM_THREADS_PER_BLOCK + tidx;
    const uint32_t total_blocks_per_scale = gridDim.y;
    const uint64_t total_threads_per_scale = total_blocks_per_scale * CUDA_MAX_NUM_THREADS_PER_BLOCK;

    // Applying scale data offsets
    input_low += scale_idx;
    input_range += scale_idx;
    dev_tmp_low += scale_idx * total_blocks_per_scale;
    dev_tmp_range += scale_idx * total_blocks_per_scale;
    dev_last_block_counter_low += scale_idx;
    dev_last_block_counter_range += scale_idx;
    grad_input_low += scale_idx;
    grad_input_range += scale_idx;

    const size_t offset_for_scaled_quantized_elements = scale_idx * elements_per_scale;
    input += offset_for_scaled_quantized_elements;
    grad_input += offset_for_scaled_quantized_elements;
    grad_output += offset_for_scaled_quantized_elements;

    scalar_accum_t per_thread_grad_sum_range = 0, per_thread_grad_sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low);
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);
    for (size_t i = per_scale_tidx; i < elements_per_scale; i += total_threads_per_scale) {
        fakeQuantize<scalar_t>(&output, (input + i), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + i), &val_grad_input_low, &val_grad_input_range, (grad_output + i),
                 (input + i), &output, range_low, range_high, reverted_range, alpha);
        per_thread_grad_sum_range += val_grad_input_range;
        per_thread_grad_sum_low += val_grad_input_low;
    }

    __shared__ scalar_accum_t  sh_grad_range[CUDA_MAX_NUM_THREADS_PER_BLOCK];
    __shared__ scalar_accum_t  sh_grad_low[CUDA_MAX_NUM_THREADS_PER_BLOCK];
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(sh_grad_range, per_thread_grad_sum_range, tidx, per_scale_block_idx, dev_tmp_range, dev_last_block_counter_range, grad_input_range, total_blocks_per_scale);
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(sh_grad_low, per_thread_grad_sum_low, tidx, per_scale_block_idx, dev_tmp_low, dev_last_block_counter_low, grad_input_low, total_blocks_per_scale);
}


template <typename scalar_t, typename scalar_accum_t>
__global__ void q_scale_per_group_cuda_backward_kernel(
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
        const size_t elements_per_scale,
        const size_t group_size,
        const size_t num_groups_per_channel) {

    const uint16_t tidx = threadIdx.x;
    const uint32_t scale_idx = blockIdx.x;
    const uint32_t per_scale_block_idx = blockIdx.y;

    const uint64_t per_scale_tidx = per_scale_block_idx * CUDA_MAX_NUM_THREADS_PER_BLOCK + tidx;
    const uint32_t total_blocks_per_scale = gridDim.y;
    const uint64_t total_threads_per_scale = total_blocks_per_scale * CUDA_MAX_NUM_THREADS_PER_BLOCK;

    // For per-group quantization, we need to map scale_idx to the correct group
    // scale_idx maps to [channel, group_within_channel] in the reshaped tensor [Cout, Cin//GS, GS]
    const size_t channel_idx = scale_idx / num_groups_per_channel;
    const size_t group_idx_in_channel = scale_idx % num_groups_per_channel;

    // Applying scale data offsets
    input_low += scale_idx;
    input_range += scale_idx;
    dev_tmp_low += scale_idx * total_blocks_per_scale;
    dev_tmp_range += scale_idx * total_blocks_per_scale;
    dev_last_block_counter_low += scale_idx;
    dev_last_block_counter_range += scale_idx;
    grad_input_low += scale_idx;
    grad_input_range += scale_idx;

    // Calculate offset for the specific group within the channel
    // Original tensor shape: [Cout, Cin], reshaped conceptually to [Cout, Cin//GS, GS]
    // We need to find the starting position of this group in the original flattened tensor
    const size_t channel_size = elements_per_scale * num_groups_per_channel; // Total elements per channel (Cin)
    const size_t group_start_in_channel = group_idx_in_channel * group_size;
    const size_t offset_for_scaled_quantized_elements = channel_idx * channel_size + group_start_in_channel;

    input += offset_for_scaled_quantized_elements;
    grad_input += offset_for_scaled_quantized_elements;
    grad_output += offset_for_scaled_quantized_elements;

    scalar_accum_t per_thread_grad_sum_range = 0, per_thread_grad_sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low);
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);

    // Process elements within this group (elements_per_scale should be group_size for per-group)
    for (size_t i = per_scale_tidx; i < elements_per_scale; i += total_threads_per_scale) {
        fakeQuantize<scalar_t>(&output, (input + i), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + i), &val_grad_input_low, &val_grad_input_range, (grad_output + i),
                 (input + i), &output, range_low, range_high, reverted_range, alpha);
        per_thread_grad_sum_range += val_grad_input_range;
        per_thread_grad_sum_low += val_grad_input_low;
    }

    __shared__ scalar_accum_t  sh_grad_range[CUDA_MAX_NUM_THREADS_PER_BLOCK];
    __shared__ scalar_accum_t  sh_grad_low[CUDA_MAX_NUM_THREADS_PER_BLOCK];
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(sh_grad_range, per_thread_grad_sum_range, tidx, per_scale_block_idx, dev_tmp_range, dev_last_block_counter_range, grad_input_range, total_blocks_per_scale);
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(sh_grad_low, per_thread_grad_sum_low, tidx, per_scale_block_idx, dev_tmp_low, dev_last_block_counter_low, grad_input_low, total_blocks_per_scale);
}

// OPTIMIZED PER-GROUP KERNEL: Process multiple groups per block to reduce grid size
template <typename scalar_t, typename scalar_accum_t>
__global__ void q_scale_per_group_cuda_backward_kernel_optimized(
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

    // KEY OPTIMIZATION: Process multiple groups per block to reduce scheduling overhead
    const int GROUPS_PER_BLOCK = 4;  // Tunable: 4x reduction in grid size

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

template <typename scalar_t, typename scalar_accum_t>
__global__ void q_scale_per_activation_channel_cuda_backward_kernel(
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
        const int64_t total_elements_per_scale,
        const int64_t contiguous_elements_per_scale,
        const int64_t scale_count,
        const int64_t leading_channel_offset) {
    const uint16_t tidx = threadIdx.x;
    const uint32_t scale_idx = blockIdx.x;
    const uint32_t per_scale_block_idx = blockIdx.y;

    const uint64_t per_scale_tidx = per_scale_block_idx * CUDA_MAX_NUM_THREADS_PER_BLOCK + tidx;
    const uint32_t total_blocks_per_scale = gridDim.y;
    const uint64_t total_threads_per_scale = total_blocks_per_scale * CUDA_MAX_NUM_THREADS_PER_BLOCK;

    // Applying scale data offsets
    input_low += scale_idx;
    input_range += scale_idx;
    dev_tmp_low += scale_idx * total_blocks_per_scale;
    dev_tmp_range += scale_idx * total_blocks_per_scale;
    dev_last_block_counter_low += scale_idx;
    dev_last_block_counter_range += scale_idx;
    grad_input_low += scale_idx;
    grad_input_range += scale_idx;

    scalar_accum_t per_thread_grad_sum_range = 0, per_thread_grad_sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low);
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);


    // The blocks of values belonging to one and the same scale here are interleaved with a period
    // equal to contiguous_elements_per_scale. Will apply an offset to the beginning of the first
    // block of values belonging to the current scale of the thread block, and then, in the for loop, map
    // a contiguously changing loop iteration index into a value-block-skipping offset calculation pattern.

    const size_t initial_offset = scale_idx * contiguous_elements_per_scale;
    input += initial_offset;
    grad_input += initial_offset;
    grad_output += initial_offset;


    for (uint64_t i = per_scale_tidx; i < total_elements_per_scale; i += total_threads_per_scale) {
        size_t additional_offset = (i / contiguous_elements_per_scale) * leading_channel_offset + (i % contiguous_elements_per_scale);
        fakeQuantize<scalar_t>(&output, (input + additional_offset), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + additional_offset), &val_grad_input_low, &val_grad_input_range, (grad_output + additional_offset),
                 (input + additional_offset), &output, range_low, range_high, reverted_range, alpha);
        per_thread_grad_sum_range += val_grad_input_range;
        per_thread_grad_sum_low += val_grad_input_low;
    }

    __shared__ scalar_accum_t sh_grad_range[CUDA_MAX_NUM_THREADS_PER_BLOCK];
    __shared__ scalar_accum_t sh_grad_low[CUDA_MAX_NUM_THREADS_PER_BLOCK];
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(sh_grad_range, per_thread_grad_sum_range, tidx, per_scale_block_idx, dev_tmp_range, dev_last_block_counter_range, grad_input_range, total_blocks_per_scale);
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(sh_grad_low, per_thread_grad_sum_low, tidx, per_scale_block_idx, dev_tmp_low, dev_last_block_counter_low, grad_input_low, total_blocks_per_scale);
}


}

at::Tensor q_cuda_forward(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels) {
    at::DeviceGuard guard(input.device());
    const auto quantized_elements_count = input.numel();

    // Detect scale type automatically based on tensor shapes
    // Per-group: weight_shape: [Cout, Cin] -> [Cout, Cin//GS, GS], scale_shape: [Cout, Cin//GS, 1]
    // Per-channel: weight_shape: [Cout, Cin], scale_shape: [Cout, 1] or [1, Cin]
    ScaleType scale_type = get_scale_type(input, input_low, input_range);

    uint64_t contiguous_elements_per_scale = 0;
    uint64_t scale_count = input_range.numel();
    switch (scale_type)
    {
        case ScaleType::PER_ACTIVATION_CHANNEL:
            // Scale count should be equal to 1-st input tensor dimension
            contiguous_elements_per_scale = quantized_elements_count / (input.size(0) * scale_count);
            break;
        case ScaleType::PER_WEIGHT_CHANNEL:
        case ScaleType::PER_GROUP:
            // Scale count should be equal to 0-th input tensor dimension
            contiguous_elements_per_scale = quantized_elements_count / scale_count;
            break;
        default:
            contiguous_elements_per_scale = quantized_elements_count;
            break;
    }


    auto output = at::empty_like(input);

    PROFILE(DISPATCH_TENSOR_DATA_TYPES(input.scalar_type(), "q_cuda_forward", ([&] {
          q_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(quantized_elements_count), CUDA_MAX_NUM_THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
              output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              input_low.data_ptr<scalar_t>(),
              input_range.data_ptr<scalar_t>(),
              levels,
              quantized_elements_count,
              contiguous_elements_per_scale,
              scale_count);
        }));)

    return output;
}


std::vector<at::Tensor> q_single_scale_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    at::DeviceGuard guard(input.device());
    const auto size = input.numel();
    auto grad_input = at::empty_like(grad_output);

    auto grad_input_range = at::empty({1}, grad_output.options());
    auto grad_input_low = at::empty({1}, grad_output.options());

    auto grid_size = std::min(GET_BLOCKS(size), CUDA_BLOCKS_PER_GRID_FOR_UNIFORM_ELTWISE);
    auto accum_options = get_accum_options(grad_output.options());

    auto dev_tmp_range = at::empty({grid_size}, accum_options);
    auto dev_tmp_low = at::empty({grid_size}, accum_options);
    auto dev_last_block_counter_range = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));

    PROFILE(DISPATCH_TENSOR_DATA_TYPES(input.scalar_type(), "q_single_scale_cuda_backward", ([&] {
      using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
      q_single_scale_cuda_backward_kernel<scalar_t, scalar_accum_t><<<grid_size, CUDA_MAX_NUM_THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_input.data_ptr<scalar_t>(),
          grad_input_low.data_ptr<scalar_t>(),
          grad_input_range.data_ptr<scalar_t>(),
          dev_tmp_range.data_ptr<scalar_accum_t>(),
          dev_tmp_low.data_ptr<scalar_accum_t>(),
          dev_last_block_counter_range.data_ptr<int32_t>(),
          dev_last_block_counter_low.data_ptr<int32_t>(),
          grad_output.data_ptr<scalar_t>(),
          input.data_ptr<scalar_t>(),
          input_low.data_ptr<scalar_t>(),
          input_range.data_ptr<scalar_t>(),
          levels,
          level_low,
          level_high,
          size);
    }));)

    return {grad_input, grad_input_low, grad_input_range};
}



std::vector<at::Tensor> q_scale_per_weight_channel_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    at::DeviceGuard guard(input.device());
    const auto scale_count = input_range.size(0);
    const auto elements_per_scale = input.numel() / scale_count;

    auto grad_input = at::empty_like(grad_output);

    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    auto accum_options = get_accum_options(grad_output.options());
    dim3 grid_size = get_2d_grid_size_for_per_channel(scale_count);
    auto dev_tmp_range = at::zeros({grid_size.x, grid_size.y}, accum_options);
    auto dev_tmp_low = at::zeros({grid_size.x, grid_size.y}, accum_options);
    auto dev_last_block_counter_range = at::zeros({grid_size.x, 1},  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low = at::zeros({grid_size.x, 1},  at::device(grad_output.options().device()).dtype(at::kInt));

    PROFILE(DISPATCH_TENSOR_DATA_TYPES(input.scalar_type(), "q_single_scale_cuda_backward", ([&] {
              using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
              q_scale_per_weight_channel_cuda_backward_kernel<scalar_t, scalar_accum_t><<<grid_size, CUDA_MAX_NUM_THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
                  grad_input.data_ptr<scalar_t>(),
                  grad_input_low.data_ptr<scalar_t>(),
                  grad_input_range.data_ptr<scalar_t>(),
                  dev_tmp_range.data_ptr<scalar_accum_t>(),
                  dev_tmp_low.data_ptr<scalar_accum_t>(),
                  dev_last_block_counter_range.data_ptr<int32_t>(),
                  dev_last_block_counter_low.data_ptr<int32_t>(),
                  grad_output.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  input_low.data_ptr<scalar_t>(),
                  input_range.data_ptr<scalar_t>(),
                  levels,
                  level_low,
                  level_high,
                  elements_per_scale);
            }));
    )

    return {grad_input, grad_input_low, grad_input_range};
}


std::vector<at::Tensor> q_scale_per_group_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    // Per-group
    // weight shape: [Cout, Cin] should be reshaped to [Cout, Cin//GS, GS] to work with scale
    // scale shape: [Cout, Cin//GS, 1]
    // Per-channel
    // weight shape: [Cout, Cin]
    // scale shape: [Cout, 1]
    at::DeviceGuard guard(input.device());

    // For per-group quantization, we need to calculate group parameters
    // input_range has shape [Cout, Cin//GS, 1] for per-group
    // input has shape [Cout, Cin] (flattened as [Cout * Cin])
    const auto scale_count = input_range.numel(); // Total number of groups: Cout * (Cin//GS)
    const auto elements_per_scale = input.numel() / scale_count; // This should be group_size (GS)

    // Calculate group structure parameters
    const auto cout = input_range.size(0); // Number of output channels
    const auto num_groups_per_channel = input_range.size(1); // Cin//GS
    const auto group_size = elements_per_scale; // GS

    auto grad_input = at::empty_like(grad_output);

    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    auto accum_options = get_accum_options(grad_output.options());
    dim3 grid_size = get_2d_grid_size_for_per_channel(scale_count);
    auto dev_tmp_range = at::zeros({grid_size.x, grid_size.y}, accum_options);
    auto dev_tmp_low = at::zeros({grid_size.x, grid_size.y}, accum_options);
    auto dev_last_block_counter_range = at::zeros({grid_size.x, 1},  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low = at::zeros({grid_size.x, 1},  at::device(grad_output.options().device()).dtype(at::kInt));

    PROFILE(DISPATCH_TENSOR_DATA_TYPES(input.scalar_type(), "q_scale_per_group_cuda_backward", ([&] {
              using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
              q_scale_per_group_cuda_backward_kernel<scalar_t, scalar_accum_t><<<grid_size, CUDA_MAX_NUM_THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
                  grad_input.data_ptr<scalar_t>(),
                  grad_input_low.data_ptr<scalar_t>(),
                  grad_input_range.data_ptr<scalar_t>(),
                  dev_tmp_range.data_ptr<scalar_accum_t>(),
                  dev_tmp_low.data_ptr<scalar_accum_t>(),
                  dev_last_block_counter_range.data_ptr<int32_t>(),
                  dev_last_block_counter_low.data_ptr<int32_t>(),
                  grad_output.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  input_low.data_ptr<scalar_t>(),
                  input_range.data_ptr<scalar_t>(),
                  levels,
                  level_low,
                  level_high,
                  elements_per_scale,
                  group_size,
                  num_groups_per_channel);
            }));
    )

    return {grad_input, grad_input_low, grad_input_range};
}

// OPTIMIZED PER-GROUP BACKWARD: Dramatically reduced grid size for better performance
std::vector<at::Tensor> q_scale_per_group_cuda_backward_optimized(at::Tensor grad_output,
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

    PROFILE(DISPATCH_TENSOR_DATA_TYPES(input.scalar_type(), "q_scale_per_group_cuda_backward_optimized", ([&] {
                using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
                q_scale_per_group_cuda_backward_kernel_optimized<scalar_t, scalar_accum_t><<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    )

    return {grad_input, grad_input_low, grad_input_range};
}



std::vector<at::Tensor> q_scale_per_activation_channel_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    at::DeviceGuard guard(input.device());
    const auto scale_count = input_range.size(1);
    const auto total_elements_per_scale = input.numel() / scale_count;
    const auto contiguous_elements_per_scale = input.numel() / (scale_count * input.size(0));
    const auto leading_channel_offset = input.numel() / input.size(0);

    auto grad_input = at::empty_like(grad_output);

    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    auto accum_options = get_accum_options(grad_output.options());
    dim3 grid_size = get_2d_grid_size_for_per_channel(scale_count);
    auto dev_tmp_range = at::zeros({grid_size.x, grid_size.y}, accum_options);
    auto dev_tmp_low = at::zeros({grid_size.x, grid_size.y}, accum_options);
    auto dev_last_block_counter_range = at::zeros({grid_size.x, 1},  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low = at::zeros({grid_size.x, 1},  at::device(grad_output.options().device()).dtype(at::kInt));

    PROFILE(
        DISPATCH_TENSOR_DATA_TYPES(input.scalar_type(), "q_scale_per_activation_channel_cuda_backward", ([&] {
          using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
          q_scale_per_activation_channel_cuda_backward_kernel<scalar_t, scalar_accum_t><<<grid_size, CUDA_MAX_NUM_THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input.data_ptr<scalar_t>(),
              grad_input_low.data_ptr<scalar_t>(),
              grad_input_range.data_ptr<scalar_t>(),
              dev_tmp_range.data_ptr<scalar_accum_t>(),
              dev_tmp_low.data_ptr<scalar_accum_t>(),
              dev_last_block_counter_range.data_ptr<int32_t>(),
              dev_last_block_counter_low.data_ptr<int32_t>(),
              grad_output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              input_low.data_ptr<scalar_t>(),
              input_range.data_ptr<scalar_t>(),
              levels,
              level_low,
              level_high,
              total_elements_per_scale,
              contiguous_elements_per_scale,
              scale_count,
              leading_channel_offset);
        }));
    )
    return {grad_input, grad_input_low, grad_input_range};
}

std::vector<at::Tensor> q_cuda_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    at::DeviceGuard guard(input.device());
    ScaleType scale_type = get_scale_type(input, input_low, input_range);

    switch (scale_type)
    {
        case ScaleType::PER_ACTIVATION_CHANNEL:
            return q_scale_per_activation_channel_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
        case ScaleType::PER_WEIGHT_CHANNEL:
            return q_scale_per_weight_channel_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
        case ScaleType::PER_GROUP:
            return q_scale_per_group_cuda_backward_optimized(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
        case ScaleType::SINGLE_SCALE:
        default:
            return q_single_scale_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
    }
}
