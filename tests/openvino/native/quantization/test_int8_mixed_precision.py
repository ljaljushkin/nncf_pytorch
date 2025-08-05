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

try:
    import openvino as ov
    from openvino.runtime import Type
    from openvino.runtime import opset13 as ops

    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

import nncf
from nncf import BackupMode
from nncf import CompressWeightsMode
from nncf import Dataset
from nncf import SensitivityMetric
from nncf.quantization import compress_weights


@pytest.mark.skipif(not OPENVINO_AVAILABLE, reason="OpenVINO not available")
class TestINT8MixedPrecisionOpenVINO:
    """Test suite for INT8 mixed precision support with OpenVINO backend."""

    def create_simple_model(self) -> ov.Model:
        """Create a simple OpenVINO model for testing."""
        input_node = ops.parameter([1, 64], Type.f32, name="input")

        # Create multiple matmul layers
        weights1 = ops.constant([[1.0] * 128] * 64, Type.f32)
        matmul1 = ops.matmul(input_node, weights1, transpose_a=False, transpose_b=False, name="matmul1")

        weights2 = ops.constant([[1.0] * 128] * 128, Type.f32)
        matmul2 = ops.matmul(matmul1, weights2, transpose_a=False, transpose_b=False, name="matmul2")

        weights3 = ops.constant([[1.0] * 64] * 128, Type.f32)
        matmul3 = ops.matmul(matmul2, weights3, transpose_a=False, transpose_b=False, name="matmul3")

        return ov.Model([matmul3], [input_node], "test_model")

    def create_sample_dataset(self):
        """Create a simple dataset for testing."""

        def data_gen():
            for _ in range(10):
                yield {"input": [[1.0] * 64]}

        return Dataset(data_gen)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("ratio", [0.8, 0.5, 0.2])
    def test_openvino_int8_mixed_precision_with_ratio_supported(self, mode, ratio):
        """Test that INT8 modes support ratio != 1 for mixed precision with OpenVINO."""
        model = self.create_simple_model()

        # This should not raise an exception
        compressed_model = compress_weights(model, mode=mode, ratio=ratio, backup_mode=BackupMode.NONE)
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_openvino_int8_mixed_precision_with_backup_mode_none(self, mode):
        """Test that INT8 modes support backup_mode=NONE for mixed precision with OpenVINO."""
        model = self.create_simple_model()

        # This should not raise an exception
        compressed_model = compress_weights(model, mode=mode, ratio=0.7, backup_mode=BackupMode.NONE)
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("backup_mode", [BackupMode.INT8_SYM, BackupMode.INT8_ASYM, BackupMode.NONE])
    def test_openvino_int8_mixed_precision_with_different_backup_modes(self, mode, backup_mode):
        """Test that INT8 modes support various backup modes in mixed precision with OpenVINO."""
        model = self.create_simple_model()

        # This should not raise an exception
        compressed_model = compress_weights(model, mode=mode, ratio=0.6, backup_mode=backup_mode)
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
    def test_openvino_int8_mixed_precision_with_dataset_and_sensitivity_metrics(self, mode, sensitivity_metric):
        """Test that INT8 modes support dataset-based mixed precision with different sensitivity metrics."""
        model = self.create_simple_model()
        dataset = self.create_sample_dataset()

        # This should not raise an exception
        compressed_model = compress_weights(
            model,
            mode=mode,
            ratio=0.5,
            dataset=dataset,
            sensitivity_metric=sensitivity_metric,
            backup_mode=BackupMode.NONE,
        )
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_openvino_int8_mixed_precision_with_weight_quantization_error_metric(self, mode):
        """Test that INT8 modes support WEIGHT_QUANTIZATION_ERROR metric without dataset."""
        model = self.create_simple_model()

        # This should not raise an exception
        compressed_model = compress_weights(
            model,
            mode=mode,
            ratio=0.4,
            sensitivity_metric=SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
            backup_mode=BackupMode.NONE,
        )
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_openvino_int8_awq_still_unsupported(self, mode):
        """Test that AWQ is still not supported for INT8 modes with OpenVINO."""
        model = self.create_simple_model()

        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*awq"):
            compress_weights(model, mode=mode, awq=True)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_openvino_int8_scale_estimation_still_unsupported(self, mode):
        """Test that scale estimation is still not supported for INT8 modes with OpenVINO."""
        model = self.create_simple_model()

        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*scale_estimation"):
            compress_weights(model, mode=mode, scale_estimation=True)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_openvino_int8_group_size_still_unsupported(self, mode):
        """Test that custom group_size is still not supported for INT8 modes with OpenVINO."""
        model = self.create_simple_model()

        with pytest.raises(nncf.ParameterNotSupportedError, match="Default values.*group_size.*cannot be overridden"):
            compress_weights(model, mode=mode, group_size=128)

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_openvino_int8_mixed_precision_requires_dataset_for_data_aware_metrics(self, mode):
        """Test that INT8 modes require dataset when using data-aware sensitivity metrics with OpenVINO."""
        model = self.create_simple_model()

        # This should raise an exception because dataset is None but sensitivity metric requires data
        with pytest.raises(nncf.ValidationError, match="Mixed precision selection.*requires a dataset"):
            compress_weights(
                model,
                mode=mode,
                ratio=0.5,
                dataset=None,  # No dataset provided
                sensitivity_metric=SensitivityMetric.MAX_ACTIVATION_VARIANCE,  # Data-aware metric
                backup_mode=BackupMode.NONE,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_openvino_int8_mixed_precision_ratio_boundary_values(self, mode):
        """Test INT8 mixed precision with boundary ratio values with OpenVINO."""
        model = self.create_simple_model()

        # Test ratio = 0 (all backup mode)
        compressed_model = compress_weights(model, mode=mode, ratio=0.0, backup_mode=BackupMode.NONE)
        assert compressed_model is not None

        # Test ratio close to 1 but not exactly 1
        compressed_model = compress_weights(model, mode=mode, ratio=0.99, backup_mode=BackupMode.NONE)
        assert compressed_model is not None

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_openvino_int8_mixed_precision_invalid_ratio_values(self, mode):
        """Test that invalid ratio values still raise errors with OpenVINO."""
        model = self.create_simple_model()

        # Test ratio > 1
        with pytest.raises(nncf.ValidationError, match="ratio should be between 0 and 1"):
            compress_weights(model, mode=mode, ratio=1.5, backup_mode=BackupMode.NONE)

        # Test negative ratio
        with pytest.raises(nncf.ValidationError, match="ratio should be between 0 and 1"):
            compress_weights(model, mode=mode, ratio=-0.1, backup_mode=BackupMode.NONE)
