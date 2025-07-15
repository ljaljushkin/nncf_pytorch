#!/usr/bin/env python3
import torch
import triton
import triton.language as tl
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta, read_stride, read_shape

@triton.jit
def debug_coordinate_kernel(
    input_ptr,
    input_meta,
    grad_low_ptr,
    grad_low_meta,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Debug kernel to understand coordinate calculation"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Read input shape
    input_s0, input_s1, input_s2, input_s3 = read_shape(input_meta)
    input_elements = input_s0 * input_s1 * input_s2 * input_s3
    
    # Read grad_low shape
    grad_low_s0, grad_low_s1, grad_low_s2, grad_low_s3 = read_shape(grad_low_meta)
    grad_low_st0, grad_low_st1, grad_low_st2, grad_low_st3 = read_stride(grad_low_meta)
    
    # Convert linear offsets to 4D coordinates
    tmp = offsets
    i3 = tmp % input_s3
    tmp //= input_s3
    i2 = tmp % input_s2
    tmp //= input_s2
    i1 = tmp % input_s1
    tmp //= input_s1
    i0 = tmp % input_s0
    
    # Calculate corresponding output coordinates for grad_low (reduction mapping)
    lo_o0 = tl.where(grad_low_s0 == 1, 0, i0)
    lo_o1 = tl.where(grad_low_s1 == 1, 0, i1)
    lo_o2 = tl.where(grad_low_s2 == 1, 0, i2)
    lo_o3 = tl.where(grad_low_s3 == 1, 0, i3)
    
    # Calculate output offsets
    grad_low_offsets = lo_o0 * grad_low_st0 + lo_o1 * grad_low_st1 + lo_o2 * grad_low_st2 + lo_o3 * grad_low_st3
    
    # For valid threads, store debug info
    valid_mask = offsets < input_elements
    debug_value = tl.where(valid_mask, 1.0, 0.0)
    
    # Store coordinates and offsets for debugging
    # For each valid thread, we'll store information about the mapping
    if pid == 0:  # Only first block
        for i in range(BLOCK_SIZE):
            if offsets[i] < input_elements:
                # Store the mapping information
                thread_id = offsets[i]
                coord_info = i0[i] * 1000 + i1[i] * 100 + i2[i] * 10 + i3[i]  # Pack coordinates
                output_offset = grad_low_offsets[i]
                
                # Store in output array: [thread_id, coord_info, output_offset]
                if thread_id < 16:  # We have space for 16 debug entries
                    tl.store(output_ptr + thread_id * 3 + 0, thread_id.to(tl.float32))
                    tl.store(output_ptr + thread_id * 3 + 1, coord_info.to(tl.float32))
                    tl.store(output_ptr + thread_id * 3 + 2, output_offset.to(tl.float32))

def test_coordinate_mapping():
    """Test coordinate mapping logic"""
    device = torch.device('cuda')
    
    # Create simple test case: 2x2x2x2 -> 1x2x1x1
    input_tensor = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)
    grad_low = torch.zeros(1, 2, 1, 1, device=device, dtype=torch.float16)
    
    # Create debug output array
    debug_output = torch.zeros(48, device=device, dtype=torch.float32)  # 16 elements * 3 values each
    
    # Get metadata
    input_meta = get_4d_tensor_meta(input_tensor)
    grad_low_meta = get_4d_tensor_meta(grad_low)
    
    print("=== Input shape and metadata ===")
    print(f"input_tensor: {input_tensor.shape}")
    print(f"grad_low: {grad_low.shape}")
    print(f"input_meta: {input_meta.cpu().numpy()}")
    print(f"grad_low_meta: {grad_low_meta.cpu().numpy()}")
    
    # Run debug kernel
    grid = lambda meta: (1,)  # Single block
    debug_coordinate_kernel[grid](
        input_tensor,
        input_meta,
        grad_low,
        grad_low_meta,
        debug_output,
        BLOCK_SIZE=256
    )
    
    # Parse debug output
    print("\n=== Coordinate mapping analysis ===")
    print("Thread | Coords(i0,i1,i2,i3) | Output_Offset")
    for i in range(16):
        thread_id = int(debug_output[i*3 + 0].item())
        coord_info = int(debug_output[i*3 + 1].item())
        output_offset = int(debug_output[i*3 + 2].item())
        
        # Unpack coordinates
        i0 = coord_info // 1000
        i1 = (coord_info // 100) % 10
        i2 = (coord_info // 10) % 10
        i3 = coord_info % 10
        
        print(f"   {thread_id:2d}  |   ({i0},{i1},{i2},{i3})     |     {output_offset}")
    
    # Count unique output offsets
    unique_offsets = set()
    for i in range(16):
        output_offset = int(debug_output[i*3 + 2].item())
        unique_offsets.add(output_offset)
    
    print(f"\nUnique output offsets: {sorted(unique_offsets)}")
    print(f"Expected output elements: {grad_low.numel()}")
    print(f"Actual unique mappings: {len(unique_offsets)}")
    
    # Check if multiple threads map to same offset
    offset_counts = {}
    for i in range(16):
        output_offset = int(debug_output[i*3 + 2].item())
        offset_counts[output_offset] = offset_counts.get(output_offset, 0) + 1
    
    print(f"\nOutput offset -> Thread count mapping:")
    for offset, count in sorted(offset_counts.items()):
        print(f"  Offset {offset}: {count} threads")
        if count > 1:
            print(f"    -> This explains {count}x over-accumulation for this offset!")

if __name__ == "__main__":
    test_coordinate_mapping()
