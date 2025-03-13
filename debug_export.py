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
from optimum.exporters.openvino.convert import export_from_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.torch.quantization.layers import BaseWeightsDecompressor

model_id = "tinyllama/tinyllama-1.1b-step-50k-105b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("dummy", return_tensors="pt").to("cuda")


def hack_nncf(model):
    model_layout = model.nncf.transformation_layout()
    transformations = model_layout.transformations
    for command in transformations:
        decompressor = command.fn
        if isinstance(decompressor, BaseWeightsDecompressor):
            decompressor.result_dtype = torch.float32


for torch_dtype in [torch.float16, torch.bfloat16]:
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
        backup_mode=nncf.BackupMode.NONE,
        ignored_scope=nncf.IgnoredScope(
            patterns=[r".*LlamaDecoderLayer\[5\]/LlamaAttention\[self_attn\]/Linear\[v_proj\].*"]
        ),
    )
    export_dir = Path("tmp_output")
    if export_dir.exists():
        shutil.rmtree(export_dir)

    hack_nncf(model)
    export_from_model(
        model.cpu(),
        export_dir,
        device="cpu",
        patch_16bit_model=True,
    )
    assert (export_dir / "openvino_model.xml").exists()
