# Copyright (c) 2025 Your Organization or Name

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
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
from datasets import load_dataset
from torch import Tensor
from torch.jit import TracerWarning
from transformers import AutoTokenizer

import nncf
from nncf.data.dataset import Dataset

warnings.filterwarnings("ignore", category=TracerWarning)


def get_wikitext2(num_samples: int, seqlen: int, tokenizer: Any) -> list[Tensor]:
    """
    Loads and processes the Wikitext-2 dataset for training.

    :param num_samples: Number of samples to generate.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer to encode the text.
    :param device: Device to move the tensors to (e.g., 'cpu' or 'cuda').
    :return: A list of tensors containing the tokenized text samples.
    """
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    limit = num_samples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    enc = tokenizer(text, return_tensors="np")
    trainloader = []
    for _ in range(num_samples):
        # Crop a sequence of tokens of length seqlen starting at a random position
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        input_ids = enc.input_ids[:, i:j]
        attention_mask = np.ones_like(input_ids)
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        trainloader.append({"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids})
    return trainloader


model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
calib_loader = get_wikitext2(num_samples=128, seqlen=32, tokenizer=tokenizer)
dataset = Dataset(calib_loader)


# model_id = "openai/gpt-oss-20b"
# model_id = "tiny-random/gpt-oss-mxfp4"
# model_dir = Path("tmp_models/gpt_oss")

model_dir = Path("tmp_models/llama3b")

bf16_dir = model_dir / "bf16"
mxfp4_dir = model_dir / "mxfp4"
int4_dir = model_dir / "int4"

# OVModelForCausalLM.from_pretrained("openai/gpt-oss-20b", quantization_config={"bits": 4}).save_pretrained(
#     "/local_ssd1/nlyalyus/tmp"
# )
# bf16_model = OVModelForCausalLM.from_pretrained(model_id, load_in_8bit=False).save_pretrained(bf16_dir)

# emulate direct OV comvertion to MXFP4
# ov_model = ov.Core().read_model(bf16_dir / "openvino_model.xml")
# compressed_model = nncf.compress_weights(
#     ov_model,
#     mode=nncf.CompressWeightsMode.MXFP4,
#     group_size=32,
#     # ignored_scope=nncf.IgnoredScope(patterns=[r".*self_attn.*", r".*router.*"]),
# )
# ov.save_model(compressed_model, mxfp4_dir / "openvino_model.xml")

# MXFP4 -> INT4 in NNCF
ov_model = ov.Core().read_model(mxfp4_dir / "openvino_model.xml")
compressed_model = nncf.compress_weights(
    ov_model,
    mode=nncf.CompressWeightsMode.INT4_SYM,
    group_size=32,
    all_layers=True,
    dataset=dataset,
    # ignored_scope=nncf.IgnoredScope(patterns=[r".*emb.*", r".*lm_head.*"]),
)
ov.save_model(compressed_model, int4_dir / "openvino_model.xml")

# 2
# compressed_model = nncf.compress_weights(
# 	ov_model,
# 	mode=nncf.CompressWeightsMode.INT4_SYM,
# 	group_size=32,
# )


# python memory_monitor.py --log-dir mxfp4_int4_decompress_in_place_compress python ../tiny_mxfp4.py
