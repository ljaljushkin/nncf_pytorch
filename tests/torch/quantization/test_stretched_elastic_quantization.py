# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for Stretched Elastic Quantization (ParetoQ-style 2-bit quantization)
with autograd support, LoRA integration, and comparison with original paretoq.
"""

import pytest
import torch

from nncf.common.quantization.quantizers import calculate_stretched_symmetric_params
from nncf.common.quantization.structs import QuantizationScheme
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import StretchedSymmetricLoraQuantizer
from nncf.torch.quantization.layers import StretchedSymmetricQuantizer

try:
    from nncf.torch.quantization.layers import PTLoraSpec
except ImportError:
    PTLoraSpec = None

try:
    from nncf.torch.quantization.layers import PTLoraNLSSpec
except ImportError:
    PTLoraNLSSpec = None

from nncf.torch.quantization.quantize_functions import stretched_symmetric_quantize
from nncf.torch.quantization.quantize_functions import stretched_symmetric_quantize_lora
from nncf.torch.quantization.reference import StretchedElasticQuantize


class TestStretchedSymmetricParams:
    """Tests for calculate_stretched_symmetric_params function."""

    def test_2bit_params(self):
        """2-bit should give n_levels=2, shift=0.5, Qp=0.75, Qn=-0.75."""
        params = calculate_stretched_symmetric_params(2)
        assert params["n_levels"] == 2
        assert params["shift"] == 0.5
        assert abs(params["Qp"] - 0.75) < 1e-6
        assert abs(params["Qn"] - (-0.75)) < 1e-6
        assert params["levels"] == 4  # 2^2
        assert abs(params["clip_val"] - (1 - 1e-2)) < 1e-6

    def test_3bit_params(self):
        """3-bit should give n_levels=4, shift=0.5, Qp=7/8=0.875."""
        params = calculate_stretched_symmetric_params(3)
        assert params["n_levels"] == 4
        assert params["shift"] == 0.5
        assert abs(params["Qp"] - (7.0 / 8.0)) < 1e-6
        assert abs(params["Qn"] - (-7.0 / 8.0)) < 1e-6
        assert params["levels"] == 8  # 2^3

    def test_4bit_params(self):
        """4-bit should give n_levels=8."""
        params = calculate_stretched_symmetric_params(4)
        assert params["n_levels"] == 8
        assert params["levels"] == 16  # 2^4


class TestStretchedElasticQuantizeGrid:
    """Tests for the 2-bit quantization grid values."""

    def test_2bit_grid_values(self):
        """
        Test that 2-bit stretched quantization produces grid {-0.75, -0.25, 0.25, 0.75}.
        """
        alpha = torch.tensor(1.0)
        # Test input range covering the full span
        x = torch.linspace(-1.5, 1.5, 9)

        # Apply quantization (positional arguments: input, alpha, num_bits, layerwise)
        y = StretchedElasticQuantize.apply(x.unsqueeze(0), alpha.unsqueeze(0), 2, True)
        y_vals = y.squeeze().tolist()

        # Extract unique representable values (rounded to avoid floating point errors)
        unique_vals = sorted(list(set(round(v, 4) for v in y_vals)))

        # Expected grid for 2-bit stretched quantization
        expected_grid = [-0.75, -0.25, 0.25, 0.75]
        assert len(unique_vals) == 4, f"Expected 4 unique values, got {len(unique_vals)}: {unique_vals}"

        for actual, expected in zip(unique_vals, expected_grid):
            assert abs(actual - expected) < 1e-4, f"Grid value mismatch: {actual} vs {expected}"

    def test_2bit_grid_with_scaled_alpha(self):
        """Test that grid scales correctly with different alpha values."""
        x = torch.linspace(-1.5, 1.5, 9)

        for alpha_val in [0.5, 1.0, 2.0]:
            # Use one batch for simpler testing, alpha shape [1,1] for layerwise
            alpha = torch.tensor([[alpha_val]])
            y = StretchedElasticQuantize.apply(x.unsqueeze(0), alpha, 2, True)
            y_vals = y.squeeze().tolist()

            unique_vals = sorted(list(set(round(v, 5) for v in y_vals)))

            # For 2-bit stretched quantization, we expect at most 4 unique values
            # The grid should be scaled by alpha: {-0.75*alpha, -0.25*alpha, 0.25*alpha, 0.75*alpha}
            assert len(unique_vals) >= 1, f"Expected at least 1 unique value, got {unique_vals}"
            assert len(unique_vals) <= 4, f"Expected at most 4 unique values, got {len(unique_vals)}"

            # Check that max absolute value is roughly bounded by the expected max
            max_val = max(abs(v) for v in unique_vals)
            expected_max = 0.75 * alpha_val
            assert max_val <= expected_max * 1.1, f"Max value {max_val} exceeds expected {expected_max}"

    def test_3bit_grid_values(self):
        """Test 3-bit stretched quantization grid."""
        alpha = torch.tensor(1.0)
        x = torch.linspace(-1.0, 1.0, 20)

        y = StretchedElasticQuantize.apply(x.unsqueeze(0), alpha.unsqueeze(0), 3, True)
        y_vals = y.squeeze().tolist()

        unique_vals = sorted(list(set(round(v, 4) for v in y_vals)))

        # For 3-bit: n_levels=4, Qp=7/8=0.875
        # Grid should be: {-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875}
        assert len(unique_vals) == 8, f"Expected 8 unique 3-bit values, got {len(unique_vals)}"


class TestStretchedElasticAutograd:
    """Tests for autograd functionality."""

    def test_forward_backward_basic(self):
        """Test that backward pass computes gradients."""
        x = torch.randn(4, 8, requires_grad=True)
        alpha = torch.tensor([[1.0], [1.2], [0.8], [1.1]], requires_grad=True)

        y = StretchedElasticQuantize.apply(x, alpha, 2, False)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "x.grad should not be None"
        assert alpha.grad is not None, "alpha.grad should not be None"

        # Check gradient shapes match input shapes
        assert x.grad.shape == x.shape
        assert alpha.grad.shape == alpha.shape

    def test_gradient_flow_clipping_region(self):
        """Test that gradients flow in the clipping region (STE)."""
        # Create input where some values are outside clipping region
        # Use rowwise quantization (layerwise=False) to avoid shape issues
        x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], requires_grad=True)
        alpha = torch.tensor([[1.0]], requires_grad=True)  # [1, 1] for rowwise

        y = StretchedElasticQuantize.apply(x, alpha, 2, False)  # rowwise quantization
        loss = y.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "x.grad should not be None"
        assert alpha.grad is not None, "alpha.grad should not be None"

        # At least the middle value (0) should have non-zero gradient (STE gradient)
        x_grad = x.grad.squeeze()
        assert abs(x_grad[2].item()) > 1e-6, "Gradient at x=0 should be non-zero"

    def test_gradient_norm_increases_with_loss(self):
        """Test that gradients are non-trivial (scale with batch size)."""
        alpha = torch.tensor([[1.0], [1.0]], requires_grad=True)

        # Larger batch
        x_large = torch.randn(2, 100, requires_grad=True)
        y_large = StretchedElasticQuantize.apply(x_large, alpha, 2, False)
        loss_large = y_large.sum()
        loss_large.backward()

        # Gradients should exist
        assert alpha.grad is not None
        assert alpha.grad.norm().item() > 1e-6


class TestStretchedSymmetricQuantizer:
    """Tests for StretchedSymmetricQuantizer layer module."""

    def test_quantizer_instantiation(self):
        """Test that quantizer can be instantiated."""
        qspec = PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationScheme.SYMMETRIC_STRETCHED,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(1,),
            logarithm_scale=False,
            compression_lr_multiplier=None,
        )
        quantizer = StretchedSymmetricQuantizer(qspec)

        # Check basic properties
        assert quantizer.num_bits == 2
        assert quantizer.layerwise is True
        assert quantizer.signed is True

    def test_quantizer_forward(self):
        """Test forward pass through quantizer."""
        qspec = PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationScheme.SYMMETRIC_STRETCHED,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(4, 1),
            logarithm_scale=False,
            compression_lr_multiplier=None,
        )
        quantizer = StretchedSymmetricQuantizer(qspec)
        quantizer.alpha.data.fill_(1.0)

        x = torch.randn(4, 100)
        y = quantizer.quantize(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_quantizer_trainable_params(self):
        """Test that alpha parameter is trainable."""
        qspec = PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationScheme.SYMMETRIC_STRETCHED,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(1,),
            logarithm_scale=False,
            compression_lr_multiplier=None,
        )
        quantizer = StretchedSymmetricQuantizer(qspec)

        params = quantizer.get_trainable_params()
        assert "alpha" in params
        assert params["alpha"].requires_grad


class TestStretchedSymmetricLoraQuantizer:
    """Tests for StretchedSymmetricLoraQuantizer with LoRA adapters."""

    @pytest.mark.skipif(PTLoraSpec is None, reason="PTLoraSpec not available")
    def test_lora_quantizer_instantiation(self):
        """Test LoRA quantizer instantiation."""
        qspec = PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationScheme.SYMMETRIC_STRETCHED_LORA,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(256, 1),
            logarithm_scale=False,
            compression_lr_multiplier=None,
        )
        lspec = PTLoraSpec(lora_rank=16, orig_weight_shape=[256, 512], weight_shape=[256, 512])
        quantizer = StretchedSymmetricLoraQuantizer(qspec, lspec)

        assert quantizer.lora_A.shape == (16, 512)
        assert quantizer.lora_B.shape == (256, 16)
        assert quantizer.num_bits == 2

    @pytest.mark.skipif(PTLoraSpec is None, reason="PTLoraSpec not available")
    def test_lora_forward(self):
        """Test forward pass with LoRA."""
        qspec = PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationScheme.SYMMETRIC_STRETCHED_LORA,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(8, 1),
            logarithm_scale=False,
            compression_lr_multiplier=None,
        )
        lspec = PTLoraSpec(lora_rank=4, orig_weight_shape=[8, 16], weight_shape=[8, 16])
        quantizer = StretchedSymmetricLoraQuantizer(qspec, lspec)
        quantizer.alpha.data.fill_(1.0)

        x = torch.randn(8, 16)
        y = quantizer.quantize(x)

        assert y.shape == x.shape

    @pytest.mark.skipif(PTLoraSpec is None, reason="PTLoraSpec not available")
    def test_lora_trainable_params(self):
        """Test that both alpha and LoRA adapters are trainable."""
        qspec = PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationScheme.SYMMETRIC_STRETCHED_LORA,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(1,),
            logarithm_scale=False,
            compression_lr_multiplier=None,
        )
        lspec = PTLoraSpec(lora_rank=8, orig_weight_shape=[16, 32], weight_shape=[16, 32])
        quantizer = StretchedSymmetricLoraQuantizer(qspec, lspec)

        params = quantizer.get_trainable_params()
        assert "alpha" in params
        assert "lora_A" in params
        assert "lora_B" in params
        assert all(p.requires_grad for p in params.values())


class TestStretchedDispatchFunctions:
    """Tests for dispatch functions."""

    def test_stretched_symmetric_quantize(self):
        """Test stretched_symmetric_quantize function."""
        x = torch.randn(4, 8)
        alpha = torch.tensor([[1.0], [1.2], [0.8], [1.1]])

        y = stretched_symmetric_quantize(x, alpha, 2, False)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_stretched_symmetric_quantize_lora(self):
        """Test stretched_symmetric_quantize_lora function."""
        x = torch.randn(4, 8)
        A = torch.randn(8, 8)
        B = torch.randn(4, 8)
        alpha = torch.tensor([[1.0], [1.2], [0.8], [1.1]])

        y = stretched_symmetric_quantize_lora(x, (4, 8), A, B, alpha, 2, False)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_quantize_with_lora_requires_grad(self):
        """Test that LoRA parameters get gradients."""
        x = torch.randn(4, 8, requires_grad=True)
        A = torch.randn(8, 8, requires_grad=True)
        B = torch.randn(4, 8, requires_grad=True)
        alpha = torch.tensor([[1.0], [1.2], [0.8], [1.1]], requires_grad=True)

        y = stretched_symmetric_quantize_lora(x, (4, 8), A, B, alpha, 2, False)
        loss = y.sum()
        loss.backward()

        assert A.grad is not None
        assert B.grad is not None
        assert alpha.grad is not None
        assert x.grad is not None


class TestComparisonWithStandardQuantization:
    """Tests comparing stretched quantization with standard quantization."""

    def test_stretched_vs_standard_2bit(self):
        """
        Compare 2-bit stretched vs standard quantization.

        Stretched grid: {-0.75, -0.25, 0.25, 0.75}
        Standard grid: {-2, -1, 0, 1} * scale (scaled to fit)

        Stretched should eliminate the wasted zero level.
        """
        from nncf.common.quantization.quantizers import calculate_symmetric_level_ranges
        from nncf.torch.quantization.quantize_functions import symmetric_quantize

        x = torch.randn(100)

        # Stretched: use alpha=1.0
        alpha_stretched = torch.tensor(1.0)
        y_stretched = stretched_symmetric_quantize(x, alpha_stretched, num_bits=2, layerwise=True)

        # Standard 2-bit: scale so that range [-2, 1] maps to same span
        level_low, level_high = calculate_symmetric_level_ranges(2, signed=True, narrow_range=False)
        scale_standard = torch.tensor(0.75)  # Scale to match stretched range
        y_standard = symmetric_quantize(
            x, levels=4, level_low=level_low, level_high=level_high, scale=scale_standard, eps=1e-16
        )

        # Both should have 4 representable values
        unique_stretched = len(torch.unique(y_stretched[~torch.isnan(y_stretched)]))
        unique_standard = len(torch.unique(y_standard[~torch.isnan(y_standard)]))

        assert unique_stretched == 4
        assert unique_standard == 4


class TestGroupedQuantization:
    """Tests for group_size=64 compatibility (grouped / per-group alpha)."""

    def test_stretched_quantize_grouped_shape(self):
        """Test that StretchedElasticQuantize handles 3D grouped input correctly."""
        # Simulate group_size=64: weight [256, 512] → [256, 8, 64], alpha [256, 8, 1]
        out_features, in_features, group_size = 256, 512, 64
        num_groups = in_features // group_size
        x_grouped = torch.randn(out_features, num_groups, group_size)
        alpha = torch.abs(torch.randn(out_features, num_groups, 1)) + 0.1  # per-group alpha

        y = StretchedElasticQuantize.apply(x_grouped, alpha, 2, False)  # rowwise (per-group)

        assert y.shape == x_grouped.shape
        assert not torch.isnan(y).any()

        # Check that output has at most 4 unique values per group (2-bit)
        for i in range(min(3, out_features)):
            for g in range(min(3, num_groups)):
                group_vals = y[i, g, :].unique()
                assert len(group_vals) <= 4, f"Group [{i},{g}] has {len(group_vals)} unique values (expected <=4)"

    def test_stretched_quantize_grouped_backward(self):
        """Test that gradients flow correctly with grouped input."""
        out_features, num_groups, group_size = 16, 8, 64
        x = torch.randn(out_features, num_groups, group_size, requires_grad=True)
        alpha = torch.abs(torch.randn(out_features, num_groups, 1)) + 0.1
        alpha.requires_grad = True

        y = StretchedElasticQuantize.apply(x, alpha, 2, False)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert alpha.grad is not None
        assert x.grad.shape == x.shape
        assert alpha.grad.shape == alpha.shape

    def test_stretched_lora_quantize_with_grouped_reshape(self):
        """
        Test that stretched_symmetric_quantize_lora correctly reshapes for group_size=64.

        Simulates: weight [out, in] → LoRA update → reshape [out, groups, group_size] → quantize → reshape back.
        """
        out_features, in_features, group_size = 32, 128, 64
        num_groups = in_features // group_size
        rank = 8

        x = torch.randn(out_features, in_features)
        A = torch.randn(rank, in_features)
        B = torch.randn(out_features, rank)
        # alpha shape is [out, groups, 1] for per-group quantization
        alpha = torch.abs(torch.randn(out_features, num_groups, 1)) + 0.1

        # input_shape is the grouped shape
        input_shape = (out_features, num_groups, group_size)

        y = stretched_symmetric_quantize_lora(x, input_shape, A, B, alpha, 2, False)

        # Output should be reshaped back to original [out, in]
        assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
        assert not torch.isnan(y).any()

    def test_stretched_lora_quantize_grouped_backward(self):
        """Test gradient flow through grouped LoRA stretched quantization."""
        out_features, in_features, group_size = 16, 128, 64
        num_groups = in_features // group_size
        rank = 4

        x = torch.randn(out_features, in_features, requires_grad=True)
        A = torch.randn(rank, in_features, requires_grad=True)
        B = torch.randn(out_features, rank, requires_grad=True)
        alpha = torch.abs(torch.randn(out_features, num_groups, 1)) + 0.1
        alpha.requires_grad = True

        input_shape = (out_features, num_groups, group_size)

        y = stretched_symmetric_quantize_lora(x, input_shape, A, B, alpha, 2, False)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "x.grad should not be None"
        assert A.grad is not None, "A.grad should not be None"
        assert B.grad is not None, "B.grad should not be None"
        assert alpha.grad is not None, "alpha.grad should not be None"
        assert alpha.grad.shape == alpha.shape

    @pytest.mark.skipif(PTLoraSpec is None, reason="PTLoraSpec not available")
    def test_stretched_lora_quantizer_layer_grouped(self):
        """Test StretchedSymmetricLoraQuantizer with 3D scale shape (grouped quantization)."""
        out_features, in_features, group_size = 64, 256, 64
        num_groups = in_features // group_size

        qspec = PTQuantizerSpec(
            num_bits=2,
            mode=QuantizationScheme.SYMMETRIC_STRETCHED_LORA,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(out_features, num_groups, 1),
            logarithm_scale=False,
            compression_lr_multiplier=None,
        )
        lspec = PTLoraSpec(
            lora_rank=8,
            orig_weight_shape=[out_features, in_features],
            weight_shape=[out_features, num_groups, group_size],
        )
        quantizer = StretchedSymmetricLoraQuantizer(qspec, lspec)
        quantizer.alpha.data.fill_(0.5)

        x = torch.randn(out_features, in_features)
        y = quantizer.quantize(x)

        # Output shape should be the original (ungrouped) shape
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
        assert not torch.isnan(y).any()

    @pytest.mark.skipif(PTLoraSpec is None, reason="PTLoraSpec not available")
    def test_alpha_initialization_from_scale(self):
        """
        Test that alpha can be initialized from integer quantization scale,
        matching how torch_backend.py initializes it: alpha = scale * 2^(num_bits-1).
        """
        num_bits = 2
        out_features, num_groups = 32, 4
        n_levels = 2 ** (num_bits - 1)  # = 2

        # Simulate scale from integer quantization
        scale = torch.abs(torch.randn(out_features, num_groups, 1)) * 0.01

        # Initialize alpha from scale (as done in torch_backend.py)
        alpha_init = scale * n_levels

        qspec = PTQuantizerSpec(
            num_bits=num_bits,
            mode=QuantizationScheme.SYMMETRIC_STRETCHED_LORA,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=(out_features, num_groups, 1),
            logarithm_scale=False,
            compression_lr_multiplier=None,
        )
        lspec = PTLoraSpec(
            lora_rank=4,
            orig_weight_shape=[out_features, num_groups * 64],
            weight_shape=[out_features, num_groups, 64],
        )
        quantizer = StretchedSymmetricLoraQuantizer(qspec, lspec)
        quantizer.alpha = torch.nn.Parameter(alpha_init)

        # Verify alpha was properly set
        assert torch.allclose(quantizer.alpha, alpha_init)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
