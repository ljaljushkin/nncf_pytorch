#!/usr/bin/env python3

"""
Analysis of performance bottleneck in q_scale_per_group_cuda_backward kernel
"""

# Input shape: [2048, 128256] (Cout=2048, Cin=128256)
# Group size: 128
# Per-group quantization reshapes: [2048, 128256] -> [2048, 128256//128, 128] = [2048, 1002, 128]
# Scale shape: [2048, 1002, 1]

cout = 2048
cin = 128256
group_size = 128

# Calculate group parameters
num_groups_per_channel = cin // group_size  # 1002
total_groups = cout * num_groups_per_channel  # 2048 * 1002 = 2,052,096
elements_per_scale = group_size  # 128 (elements per group)

print(f"Input shape: [{cout}, {cin}]")
print(f"Group size: {group_size}")
print(f"Groups per channel: {num_groups_per_channel}")
print(f"Total groups (scale_count): {total_groups:,}")
print(f"Elements per scale: {elements_per_scale}")
print()

# CUDA Grid configuration analysis
CUDA_TARGET_SM_COUNT = 72  # RTX 2080 Ti
CUDA_TARGET_NUM_THREADS_PER_SM = 2048
CUDA_MAX_NUM_THREADS_PER_BLOCK = 1024
CUDA_MAX_GRID_SIZE_Y = 65535
CUDA_WARP_SIZE = 32
CUDA_MAX_WARPS_PER_BLOCK = CUDA_MAX_NUM_THREADS_PER_BLOCK // CUDA_WARP_SIZE


def align(num, alignment):
    return (num & ~(alignment - 1)) + alignment


def get_2d_grid_size_for_per_channel(scale_count):
    # X will correspond to scale count, Y will be determined in order to hit the thread-per-SM target
    grid_size_x = scale_count
    available_threads_per_scale = int((CUDA_TARGET_SM_COUNT * CUDA_TARGET_NUM_THREADS_PER_SM + 0.0) / grid_size_x)
    available_warps_per_scale = align(available_threads_per_scale, CUDA_WARP_SIZE) // CUDA_WARP_SIZE
    blocks_per_scale = max(1, available_warps_per_scale // CUDA_MAX_WARPS_PER_BLOCK)
    grid_size_y = min(blocks_per_scale, CUDA_MAX_GRID_SIZE_Y)

    return (grid_size_x, grid_size_y)


grid_x, grid_y = get_2d_grid_size_for_per_channel(total_groups)
print(f"Grid size: ({grid_x:,}, {grid_y})")
print(f"Total blocks: {grid_x * grid_y:,}")
print(f"Total threads: {grid_x * grid_y * CUDA_MAX_NUM_THREADS_PER_BLOCK:,}")
print()

# Calculate expected workload
print("=== WORKLOAD ANALYSIS ===")
print(f"Each block processes: {elements_per_scale} elements")
print(f"Total elements to process: {cout * cin:,}")
print(f"Work per thread (avg): {(cout * cin) / (grid_x * grid_y * CUDA_MAX_NUM_THREADS_PER_BLOCK):.2f} elements")
print()

# Problem identification
print("=== IDENTIFIED PROBLEMS ===")
print(f"1. MASSIVE GRID SIZE: {grid_x:,} blocks in X dimension")
print("   - This creates millions of blocks, causing severe scheduling overhead")
print("   - Each block only processes 128 elements (group_size)")
print("   - GPU scheduler struggles with so many tiny blocks")
print()

print("2. POOR WORK DISTRIBUTION:")
print(f"   - Total work: {cout * cin:,} elements")
print(f"   - Number of groups: {total_groups:,}")
print(f"   - Work per group: {group_size} elements")
print(f"   - This creates {total_groups:,} separate reductions!")
print()

print("3. MEMORY ACCESS PATTERN:")
print("   - Each group accesses non-contiguous memory locations")
print("   - Poor cache locality due to scattered access pattern")
print()

# Compare with per-channel
print("=== COMPARISON WITH PER-CHANNEL ===")
pc_scale_count = cout  # 2048 scales for per-channel
pc_elements_per_scale = cin  # 128256 elements per scale
pc_grid_x, pc_grid_y = get_2d_grid_size_for_per_channel(pc_scale_count)

print(f"Per-channel scale count: {pc_scale_count}")
print(f"Per-channel elements per scale: {pc_elements_per_scale:,}")
print(f"Per-channel grid size: ({pc_grid_x}, {pc_grid_y})")
print(f"Per-channel total blocks: {pc_grid_x * pc_grid_y:,}")
print(f"Reduction factor: {(grid_x * grid_y) / (pc_grid_x * pc_grid_y):.1f}x more blocks for per-group")
