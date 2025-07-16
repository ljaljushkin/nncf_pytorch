#!/usr/bin/env python3
import numpy as np
import torch

from nncf.torch.quantization.triton.reference import backward

# Test with single_scale mode
device = torch.device("cuda")

# Create test data similar to what fails
input_size = [1, 16, 64, 64]
input_tensor = torch.randn(input_size, device=device, dtype=torch.float16)
grad_output = torch.randn(input_size, device=device, dtype=torch.float16)

# For single_scale mode, input_low and input_range should have shape [1]
input_low = torch.tensor([-128.0], device=device, dtype=torch.float16)
input_range = torch.tensor([255.0], device=device, dtype=torch.float16)

print("input_tensor shape:", input_tensor.shape)
print("grad_output shape:", grad_output.shape)
print("input_low shape:", input_low.shape)
print("input_range shape:", input_range.shape)

# Test the backward function
levels = 256
level_low = 0
level_high = 255

try:
    grad_input, grad_low, grad_range = backward(
        grad_output, input_tensor, input_low, input_range, levels, level_low, level_high, is_asymmetric=True
    )

    print("grad_input shape:", grad_input.shape)
    print("grad_low shape:", grad_low.shape)
    print("grad_range shape:", grad_range.shape)
    print("grad_low value:", grad_low)
    print("grad_range value:", grad_range)

    print("SUCCESS: Backward function worked")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
