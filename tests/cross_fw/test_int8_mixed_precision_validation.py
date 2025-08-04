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

import nncf
from nncf import BackupMode
from nncf import CompressWeightsMode
from nncf import SensitivityMetric
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.algorithm import check_user_compression_configuration


class TestINT8MixedPrecisionValidation:
    """Test suite for INT8 mixed precision configuration validation."""

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("ratio", [0.0, 0.2, 0.5, 0.8, 0.99, 1.0])
    def test_int8_ratio_values_allowed(self, mode, ratio):
        """Test that various ratio values are now allowed for INT8 modes."""
        # This should not raise an exception
        check_user_compression_configuration(
            mode=mode,
            subset_size=128,
            dataset=None,
            ratio=ratio,
            group_size=None,  # Default -1
            all_layers=None,
            awq=None,
            scale_estimation=None,
            gptq=None,
            lora_correction=None,
            ignored_scope=None,
            sensitivity_metric=None,
            backup_mode=BackupMode.NONE,
            compression_format=None,
            advanced_parameters=None,
        )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("backup_mode", [BackupMode.NONE, BackupMode.INT8_SYM, BackupMode.INT8_ASYM])
    def test_int8_backup_mode_none_allowed(self, mode, backup_mode):
        """Test that backup_mode=NONE is now allowed for INT8 modes."""
        # This should not raise an exception
        check_user_compression_configuration(
            mode=mode,
            subset_size=128,
            dataset=None,
            ratio=0.5,
            group_size=None,  # Default -1
            all_layers=None,
            awq=None,
            scale_estimation=None,
            gptq=None,
            lora_correction=None,
            ignored_scope=None,
            sensitivity_metric=SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
            backup_mode=backup_mode,
            compression_format=None,
            advanced_parameters=None,
        )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_awq_still_not_allowed(self, mode):
        """Test that AWQ is still not allowed for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*awq"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=None,
                group_size=None,
                all_layers=None,
                awq=True,  # This should cause the error
                scale_estimation=None,
                gptq=None,
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_scale_estimation_still_not_allowed(self, mode):
        """Test that scale estimation is still not allowed for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*scale_estimation"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=None,
                group_size=None,
                all_layers=None,
                awq=None,
                scale_estimation=True,  # This should cause the error
                gptq=None,
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_gptq_still_not_allowed(self, mode):
        """Test that GPTQ is still not allowed for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*gptq"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=None,
                group_size=None,
                all_layers=None,
                awq=None,
                scale_estimation=None,
                gptq=True,  # This should cause the error
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_lora_correction_still_not_allowed(self, mode):
        """Test that LoRA correction is still not allowed for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*lora_correction"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=None,
                group_size=None,
                all_layers=None,
                awq=None,
                scale_estimation=None,
                gptq=None,
                lora_correction=True,  # This should cause the error
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_all_layers_still_not_allowed(self, mode):
        """Test that all_layers is still not allowed for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*all_layers"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=None,
                group_size=None,
                all_layers=True,  # This should cause the error
                awq=None,
                scale_estimation=None,
                gptq=None,
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_group_size_still_not_allowed(self, mode):
        """Test that custom group_size is still not allowed for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="Default values.*group_size.*cannot be overridden"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=None,
                group_size=128,  # This should cause the error
                all_layers=None,
                awq=None,
                scale_estimation=None,
                gptq=None,
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_statistics_path_still_not_allowed(self, mode):
        """Test that statistics_path is still not allowed for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*statistics_path"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=None,
                group_size=None,
                all_layers=None,
                awq=None,
                scale_estimation=None,
                gptq=None,
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=AdvancedCompressionParameters(statistics_path="/some/path"),
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_multiple_unsupported_options_reported(self, mode):
        """Test that multiple unsupported options are properly reported for INT8 modes."""
        with pytest.raises(nncf.ParameterNotSupportedError, match="INT8 modes do not support.*awq.*gptq.*all_layers"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=None,
                group_size=None,
                all_layers=True,  # Unsupported
                awq=True,  # Unsupported
                scale_estimation=None,
                gptq=True,  # Unsupported
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("ratio", [-0.1, 1.1, 2.0])
    def test_int8_invalid_ratio_values_rejected(self, mode, ratio):
        """Test that invalid ratio values are still rejected for INT8 modes."""
        with pytest.raises(nncf.ValidationError, match="ratio should be between 0 and 1"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,
                ratio=ratio,  # Invalid ratio
                group_size=None,
                all_layers=None,
                awq=None,
                scale_estimation=None,
                gptq=None,
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize("subset_size", [0, -1, -10])
    def test_int8_invalid_subset_size_rejected(self, mode, subset_size):
        """Test that invalid subset_size values are rejected for INT8 modes."""
        with pytest.raises(nncf.ValidationError, match="subset_size value should be positive"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=subset_size,  # Invalid subset_size
                dataset=None,
                ratio=None,
                group_size=None,
                all_layers=None,
                awq=None,
                scale_estimation=None,
                gptq=None,
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=None,
                backup_mode=None,
                compression_format=None,
                advanced_parameters=None,
            )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    @pytest.mark.parametrize(
        "sensitivity_metric",
        [
            SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
            SensitivityMetric.MAX_ACTIVATION_VARIANCE,
            SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
            SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
        ],
    )
    def test_int8_all_sensitivity_metrics_allowed(self, mode, sensitivity_metric):
        """Test that all sensitivity metrics are allowed for INT8 modes."""
        # This should not raise an exception
        check_user_compression_configuration(
            mode=mode,
            subset_size=128,
            dataset=None,
            ratio=0.5,
            group_size=None,
            all_layers=None,
            awq=None,
            scale_estimation=None,
            gptq=None,
            lora_correction=None,
            ignored_scope=None,
            sensitivity_metric=sensitivity_metric,
            backup_mode=BackupMode.NONE,
            compression_format=None,
            advanced_parameters=None,
        )

    @pytest.mark.parametrize("mode", [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM])
    def test_int8_data_aware_metric_requires_dataset(self, mode):
        """Test that data-aware sensitivity metrics require dataset for INT8 modes."""
        with pytest.raises(nncf.ValidationError, match="Mixed precision selection.*requires a dataset"):
            check_user_compression_configuration(
                mode=mode,
                subset_size=128,
                dataset=None,  # No dataset provided
                ratio=0.5,
                group_size=None,
                all_layers=None,
                awq=None,
                scale_estimation=None,
                gptq=None,
                lora_correction=None,
                ignored_scope=None,
                sensitivity_metric=SensitivityMetric.MAX_ACTIVATION_VARIANCE,  # Data-aware metric
                backup_mode=BackupMode.NONE,
                compression_format=None,
                advanced_parameters=None,
            )
