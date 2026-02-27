#ifndef _COMMON_CUDA_DEFS_CUH_
#define _COMMON_CUDA_DEFS_CUH_

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

const uint32_t CUDA_WARP_SIZE = 32;
// Hardware-specific constants - should be detected at runtime for optimal performance
// const uint32_t CUDA_TARGET_NUM_THREADS_PER_SM = 1536; // RTX 3090: 1536, RTX 2080Ti: 2048
// const uint32_t CUDA_TARGET_SM_COUNT = 82; // RTX 3090: 82, RTX 2080Ti: 68, RTX 4090: 128
const uint32_t CUDA_TARGET_NUM_THREADS_PER_SM = 2048; // RTX 3090: 1536, RTX 2080Ti: 2048
const uint32_t CUDA_TARGET_SM_COUNT = 72; // RTX 3090: 82, RTX 2080Ti: 68, RTX 4090: 128
const uint32_t CUDA_MAX_NUM_THREADS_PER_BLOCK = 1024; // Maximum for all CUDA compute capabilities up to 8.0

// Runtime hardware detection would be ideal:
// cudaDeviceProp props; cudaGetDeviceProperties(&props, 0);
// const uint32_t ACTUAL_SM_COUNT = props.multiProcessorCount;
// const uint32_t ACTUAL_THREADS_PER_SM = props.maxThreadsPerMultiProcessor;
const uint16_t CUDA_MAX_WARPS_PER_BLOCK = CUDA_MAX_NUM_THREADS_PER_BLOCK / CUDA_WARP_SIZE;
const uint32_t CUDA_BLOCKS_PER_GRID_FOR_UNIFORM_ELTWISE = CUDA_TARGET_SM_COUNT * CUDA_TARGET_NUM_THREADS_PER_SM / CUDA_MAX_NUM_THREADS_PER_BLOCK;
const uint16_t CUDA_MAX_GRID_SIZE_Y = 65535;

inline uint32_t GET_BLOCKS(const uint32_t total_required_threads) {
    return (total_required_threads + CUDA_MAX_NUM_THREADS_PER_BLOCK - 1) / CUDA_MAX_NUM_THREADS_PER_BLOCK;
}

inline c10::TensorOptions get_accum_options(const c10::TensorOptions options) {
    if (options.dtype() == c10::ScalarType::Half || options.dtype() == c10::ScalarType::BFloat16) {
        return options.dtype(c10::ScalarType::Float);
    }
    return options;
}


template<class I>
inline I align(I num, I alignment)
{
    return (num & ~(alignment - 1)) + alignment;
}

inline dim3 get_2d_grid_size_for_per_channel(const uint32_t scale_count)
{
    // // For very large scale counts (e.g., 38K+ for per-group), limit grid size to avoid over-subscription
    // const uint32_t MAX_REASONABLE_BLOCKS = CUDA_TARGET_SM_COUNT * 8; // 8 blocks per SM max

    // if (scale_count > MAX_REASONABLE_BLOCKS) {
    //     // Use 1D grid approach for large scale counts to avoid creating too many blocks
    //     uint32_t optimal_blocks = std::min(scale_count, MAX_REASONABLE_BLOCKS);
    //     return dim3(optimal_blocks, 1);
    // }

    // Original logic for reasonable scale counts
    uint32_t grid_size_x = scale_count;
    uint32_t available_threads_per_scale = static_cast<uint32_t>((CUDA_TARGET_SM_COUNT * CUDA_TARGET_NUM_THREADS_PER_SM + 0.0) / grid_size_x);
    uint32_t available_warps_per_scale = align(available_threads_per_scale, CUDA_WARP_SIZE) / CUDA_WARP_SIZE;
    uint32_t blocks_per_scale = std::max(1U, available_warps_per_scale / static_cast<uint32_t>(CUDA_MAX_WARPS_PER_BLOCK));
    uint16_t grid_size_y = std::min(blocks_per_scale, static_cast<uint32_t>(CUDA_MAX_GRID_SIZE_Y));

    return dim3(grid_size_x, grid_size_y);
}



#ifdef DO_PROFILE
#define PROFILE(CODE)                                                        \
    int iter = 10;                                                           \
    for (int i = 0; i < iter; i++) {                                         \
        CODE                                                                 \
    }                                                                        \
    cudaDeviceSynchronize();                                                 \
    auto start = std::chrono::steady_clock::now();                           \
    for (int i = 0; i < iter; i++) {                                         \
            CODE                                                             \
        }                                                                    \
    cudaDeviceSynchronize();                                                 \
    auto end = std::chrono::steady_clock::now();                             \
    std::chrono::duration<double> diff = (end - start) / iter;               \
    std::cout << "PROFILE: avg kernel runtime = " <<                         \
        std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count()   \
        << " ns" << std::endl;                                               \
    cudaError_t err = cudaGetLastError();                                    \
    if (err != cudaSuccess)                                                  \
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
#else
#define PROFILE(CODE) CODE
#endif

#define ACCUM_TYPE_FOR(SOURCE_TYPE) \
std::conditional_t<std::is_same<SOURCE_TYPE, at::Half>::value, float, \
                   std::conditional_t<std::is_same<SOURCE_TYPE, at::BFloat16>::value, float, SOURCE_TYPE>>


#endif // _COMMON_CUDA_DEFS_CUH_
