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

import shutil
from pathlib import Path

import torch
import torch.nn as nn
from optimum.exporters.openvino.convert import export_from_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf


def new_patching_on_export(model):
    is_bfloat16 = any(p.dtype == torch.bfloat16 for p in model.parameters()) or any(
        b.dtype == torch.bfloat16 for b in model.buffers()
    )
    model_dtype = torch.bfloat16 if is_bfloat16 else torch.float16

    def cast_inputs_to_weight_dtype(module, inputs):
        inputs = tuple(
            input.to(model_dtype if input.dtype in (torch.float32, torch.float64) else input.dtype) for input in inputs
        )
        return inputs

    def cast_output_to_float32(module, input, output):
        return output.to(torch.float32)

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.register_forward_hook(cast_output_to_float32)
            module.register_forward_pre_hook(cast_inputs_to_weight_dtype)
    return model


model_id = "tinyllama/tinyllama-1.1b-step-50k-105b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("dummy", return_tensors="pt").to("cuda")

for torch_dtype in [torch.bfloat16, torch.float16]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="cuda",
    )
    model = nncf.compress_weights(
        model,
        group_size=64,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        dataset=nncf.Dataset([dict(inputs)]),
    )
    export_dir = Path("tmp_output")
    if export_dir.exists():
        shutil.rmtree(export_dir)

    model = new_patching_on_export(model)
    export_from_model(
        model.cpu(),
        export_dir,
        device="cpu",
        patch_16bit_model=False,
    )
    assert (export_dir / "openvino_model.xml").exists()
