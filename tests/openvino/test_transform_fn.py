# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import openvino
import openvino.runtime as ov
import pytest
from openvino.runtime import Core

import nncf
from nncf import CompressWeightsMode
from nncf import ModelType
from tests.openvino.native.models import AWQMatmulModel
from tests.openvino.native.models import ConvModel as ModelWithMultipleInputs
from tests.openvino.native.models import LinearModel as ModelWithSingleInput

core = Core()

dataset = [
    {
        "input_0": np.zeros((1, 3, 4, 2), dtype=np.float32),
        "input_1": np.zeros((1, 3, 2, 4), dtype=np.float32),
    }
]


def single_input_transform_fn(data_item):
    return data_item["input_0"]


def multiple_inputs_transform_fn(data_item):
    return data_item["input_0"], data_item["input_1"]


def multiple_inputs_as_dict_transform_fn(data_item):
    return {
        "Input_1": data_item["input_0"],
        "Input_2": data_item["input_1"],
    }

def test_quantize_compressed_weight_simple(tmp_path):
    model = ModelWithSingleInput().ov_model
    calibration_dataset = nncf.Dataset(dataset, single_input_transform_fn)

    openvino.save_model(model, tmp_path / 'model.xml', compress_to_fp16=False)

    compressed_model = nncf.compress_weights(model, mode=CompressWeightsMode.INT8_SYM)#, group_size=2)
    openvino.save_model(compressed_model, tmp_path / 'compressed_model.xml', compress_to_fp16=False)

    # compressed_model = core.read_model(tmp_path / 'compressed_model.xml')
    quantized_model = nncf.quantize(compressed_model, calibration_dataset, model_type=ModelType.TRANSFORMER)
    openvino.save_model(quantized_model, tmp_path / 'quantized_model.xml', compress_to_fp16=False)

def test_quantize_compressed_weight_awq_model(tmp_path):
    model = AWQMatmulModel().ov_model
    calibration_dataset = nncf.Dataset([np.ones([8, 8])])

    openvino.save_model(model, tmp_path / 'model.xml', compress_to_fp16=False)

    compressed_model = nncf.compress_weights(model, mode=CompressWeightsMode.INT8_SYM)#, group_size=2)
    openvino.save_model(compressed_model, tmp_path / 'compressed_model.xml', compress_to_fp16=False)

    # compressed_model = core.read_model(tmp_path / 'compressed_model.xml')
    quantized_model = nncf.quantize(compressed_model, calibration_dataset, model_type=ModelType.TRANSFORMER)
    openvino.save_model(quantized_model, tmp_path / 'quantized_model.xml', compress_to_fp16=False)

