#!/usr/bin/env python3
import torch

# Test what PyTorch expects for scalar vs [1] gradients
print("Testing PyTorch autograd gradient shape expectations...")


# Create a simple function that takes scalar and [1] inputs
class TestFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scalar_param, single_elem_param):
        ctx.save_for_backward(input_tensor, scalar_param, single_elem_param)
        return input_tensor * scalar_param * single_elem_param

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, scalar_param, single_elem_param = ctx.saved_tensors

        # Return gradients with matching shapes
        grad_input = grad_output * scalar_param * single_elem_param
        grad_scalar = (grad_output * input_tensor * single_elem_param).sum()
        grad_single_elem = (grad_output * input_tensor * scalar_param).sum()

        print(
            f"Input shapes: input_tensor={input_tensor.shape}, scalar_param={scalar_param.shape}, single_elem_param={single_elem_param.shape}"
        )
        print(
            f"Grad shapes: grad_input={grad_input.shape}, grad_scalar={grad_scalar.shape}, grad_single_elem={grad_single_elem.shape}"
        )

        return grad_input, grad_scalar, grad_single_elem


# Test with different parameter shapes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test case 1: scalar parameter (shape [])
input_tensor = torch.randn(2, 3, device=device, requires_grad=True)
scalar_param = torch.tensor(2.0, device=device, requires_grad=True)  # shape []
single_elem_param = torch.tensor([3.0], device=device, requires_grad=True)  # shape [1]

print("\nTest case 1: scalar vs [1] parameters")
print(f"scalar_param shape: {scalar_param.shape}")
print(f"single_elem_param shape: {single_elem_param.shape}")

try:
    result = TestFunction.apply(input_tensor, scalar_param, single_elem_param)
    loss = result.sum()
    loss.backward()
    print("✓ Success - gradients computed correctly")
    print(f"scalar_param.grad shape: {scalar_param.grad.shape}")
    print(f"single_elem_param.grad shape: {single_elem_param.grad.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test case 2: both [1] parameters
input_tensor2 = torch.randn(2, 3, device=device, requires_grad=True)
param1 = torch.tensor([2.0], device=device, requires_grad=True)  # shape [1]
param2 = torch.tensor([3.0], device=device, requires_grad=True)  # shape [1]

print("\nTest case 2: both [1] parameters")
print(f"param1 shape: {param1.shape}")
print(f"param2 shape: {param2.shape}")

try:
    result = TestFunction.apply(input_tensor2, param1, param2)
    loss = result.sum()
    loss.backward()
    print("✓ Success - gradients computed correctly")
    print(f"param1.grad shape: {param1.grad.shape}")
    print(f"param2.grad shape: {param2.grad.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
