# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn as nn

import nncf
from nncf import BackupMode
from nncf import CompressWeightsMode
from nncf import Dataset
from nncf import SensitivityMetric
from nncf.quantization import compress_weights
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.torch.function_hook import wrap_model
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper


class SimpleLinearModel(nn.Module):
    """Simple model with multiple linear layers for testing mixed precision."""

    def __init__(self, n_layers=4, input_size=64, hidden_size=128):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Last layer
        self.layers.append(nn.Linear(hidden_size, input_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def simple_model():
    """Fixture that provides a simple linear model."""
    return SimpleLinearModel()


@pytest.fixture
def wrapped_model(simple_model):
    """Fixture that provides a wrapped model for NNCF."""
    dummy_input = torch.randn(1, 64)
    return GraphModelWrapper(wrap_model(simple_model), example_input=dummy_input)


@pytest.fixture
def sample_dataset():
    """Fixture that provides a small dataset for testing."""

    def data_gen():
        for _ in range(10):
            yield torch.randn(1, 64)

    return Dataset(data_gen)


class TestINT8MixedPrecisionSupport:
    """Test suite for INT8 mixed precision support."""

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("ratio", [0.8, 0.5, 0.2])
    def test_int8_mixed_precision_with_ratio_supported(self, wrapped_model, mode, ratio):
        """Test that INT8 modes now support ratio != 1 for mixed precision."""
        # This should not raise an exception
        compressed_model = compress_weights(wrapped_model, mode=mode, ratio=ratio, backup_mode=BackupMode.NONE)
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_mixed_precision_with_backup_mode_none(self, wrapped_model, mode):
        """Test that INT8 modes support backup_mode=NONE for mixed precision."""
        # This should not raise an exception
        compressed_model = compress_weights(wrapped_model, mode=mode, ratio=0.7, backup_mode=BackupMode.NONE)
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("backup_mode", [BackupMode.INT8_SYM, BackupMode.INT8_ASYM, BackupMode.NONE])
    def test_int8_mixed_precision_with_different_backup_modes(self, wrapped_model, mode, backup_mode):
        """Test that INT8 modes support various backup modes in mixed precision."""
        # This should not raise an exception
        compressed_model = compress_weights(wrapped_model, mode=mode, ratio=0.6, backup_mode=backup_mode)
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize(
        "sensitivity_metric",
        [
            SensitivityMetric.MAX_ACTIVATION_VARIANCE,
            SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
            SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
        ],
    )
    def test_int8_mixed_precision_with_dataset_and_sensitivity_metrics(
        self, wrapped_model, sample_dataset, mode, sensitivity_metric
    ):
        """Test that INT8 modes support dataset-based mixed precision with different sensitivity metrics."""
        # This should not raise an exception
        compressed_model = compress_weights(
            wrapped_model,
            mode=mode,
            ratio=0.5,
            dataset=sample_dataset,
            sensitivity_metric=sensitivity_metric,
            backup_mode=BackupMode.NONE,
        )
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_mixed_precision_with_weight_quantization_error_metric(self, wrapped_model, mode):
        """Test that INT8 modes support WEIGHT_QUANTIZATION_ERROR metric without dataset."""
        # This should not raise an exception
        compressed_model = compress_weights(
            wrapped_model,
            mode=mode,
            ratio=0.4,
            sensitivity_metric=SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
            backup_mode=BackupMode.NONE,
        )
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_mixed_precision_requires_dataset_for_data_aware_metrics(self, wrapped_model, mode):
        """Test that INT8 modes require dataset when using data-aware sensitivity metrics."""
        # This should raise an exception because dataset is None but sensitivity metric requires data
        with pytest.raises(nncf.ValidationError, match="Mixed precision selection.*requires a dataset"):
            compress_weights(
                wrapped_model,
                mode=mode,
                ratio=0.5,
                dataset=None,  # No dataset provided
                sensitivity_metric=SensitivityMetric.MAX_ACTIVATION_VARIANCE,  # Data-aware metric
                backup_mode=BackupMode.NONE,
            )


class TestINT8StillUnsupportedFeatures:
    """Test suite to ensure certain features are still not supported for INT8 modes."""

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_awq_still_unsupported(self, wrapped_model, mode):
        """Test that AWQ is still not supported for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*awq"):
            compress_weights(wrapped_model, mode=mode, awq=True)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_scale_estimation_still_unsupported(self, wrapped_model, mode):
        """Test that scale estimation is still not supported for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*scale_estimation"):
            compress_weights(wrapped_model, mode=mode, scale_estimation=True)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_gptq_still_unsupported(self, wrapped_model, mode):
        """Test that GPTQ is still not supported for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*gptq"):
            compress_weights(wrapped_model, mode=mode, gptq=True)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_lora_correction_still_unsupported(self, wrapped_model, mode):
        """Test that LoRA correction is still not supported for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*lora_correction"):
            compress_weights(wrapped_model, mode=mode, lora_correction=True)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_all_layers_still_unsupported(self, wrapped_model, mode):
        """Test that all_layers is still not supported for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*all_layers"):
            compress_weights(wrapped_model, mode=mode, all_layers=True)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_group_size_still_unsupported(self, wrapped_model, mode):
        """Test that custom group_size is still not supported for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="Default values.*group_size.*cannot be overridden"):
            compress_weights(wrapped_model, mode=mode, group_size=128)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_statistics_path_still_unsupported(self, wrapped_model, mode):
        """Test that statistics_path is still not supported for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*statistics_path"):
            compress_weights(
                wrapped_model,
                mode=mode,
                advanced_parameters=AdvancedCompressionParameters(statistics_path="/some/path"),
            )


class TestINT8MixedPrecisionCombinations:
    """Test suite for various combinations of INT8 mixed precision parameters."""

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("ratio", [0.9, 0.7, 0.3])
    @pytest.mark.parametrize("backup_mode", [BackupMode.NONE, BackupMode.INT8_SYM, BackupMode.INT8_ASYM])
    def test_int8_mixed_precision_ratio_backup_combinations(self, wrapped_model, mode, ratio, backup_mode):
        """Test various combinations of ratio and backup_mode for INT8 mixed precision."""
        # This should not raise an exception
        compressed_model = compress_weights(wrapped_model, mode=mode, ratio=ratio, backup_mode=backup_mode)
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_mixed_precision_with_dataset_no_sensitivity_metric(self, wrapped_model, sample_dataset, mode):
        """Test INT8 mixed precision with dataset but default sensitivity metric."""
        # Should work - default sensitivity metric should be selected appropriately
        compressed_model = compress_weights(
            wrapped_model, mode=mode, ratio=0.6, dataset=sample_dataset, backup_mode=BackupMode.NONE
        )
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_mixed_precision_no_dataset_weight_quantization_error_metric(self, wrapped_model, mode):
        """Test INT8 mixed precision without dataset using WEIGHT_QUANTIZATION_ERROR metric."""
        # Should work - WEIGHT_QUANTIZATION_ERROR doesn't require dataset
        compressed_model = compress_weights(
            wrapped_model,
            mode=mode,
            ratio=0.8,
            sensitivity_metric=SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
            backup_mode=BackupMode.NONE,
        )
        assert compressed_model is not None


class TestINT8MixedPrecisionEdgeCases:
    """Test suite for edge cases in INT8 mixed precision."""

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_mixed_precision_ratio_boundary_values(self, wrapped_model, mode):
        """Test INT8 mixed precision with boundary ratio values."""
        # Test ratio = 0 (all backup mode)
        compressed_model = compress_weights(wrapped_model, mode=mode, ratio=0.0, backup_mode=BackupMode.NONE)
        assert compressed_model is not None

        # Test ratio close to 1 but not exactly 1
        compressed_model = compress_weights(wrapped_model, mode=mode, ratio=0.99, backup_mode=BackupMode.NONE)
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_mixed_precision_invalid_ratio_values(self, wrapped_model, mode):
        """Test that invalid ratio values still raise errors."""
        # Test ratio > 1
        with pytest.raises(nncf.ValidationError, match="ratio should be between 0 and 1"):
            compress_weights(wrapped_model, mode=mode, ratio=1.5, backup_mode=BackupMode.NONE)

        # Test negative ratio
        with pytest.raises(nncf.ValidationError, match="ratio should be between 0 and 1"):
            compress_weights(wrapped_model, mode=mode, ratio=-0.1, backup_mode=BackupMode.NONE)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_mixed_precision_multiple_unsupported_params(self, wrapped_model, mode):
        """Test that multiple unsupported parameters are reported correctly."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*awq.*gptq"):
            compress_weights(wrapped_model, mode=mode, awq=True, gptq=True, ratio=0.5, backup_mode=BackupMode.NONE)
