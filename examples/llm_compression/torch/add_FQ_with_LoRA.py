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

import argparse
import random
import sys
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.common.logging.logger import set_log_file


def save_checkpoint(wrapped_model, ckpt_dir):
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    wrapped_model = wrapped_model.cpu()
    nncf_state_dict = wrapped_model.nncf.state_dict()
    nncf_config = wrapped_model.nncf.get_config()
    ckpt_path = ckpt_dir / "nncf_checkpoint.pth"
    print(f"Saving ckpt to: {ckpt_path}")
    torch.save(
        {
            "nncf_state_dict": nncf_state_dict,
            "nncf_config": nncf_config,
        },
        ckpt_path,
    )


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(add_help=True)
# Model params
parser.add_argument("-m", "--model_id")
args = parser.parse_args()

set_seed(42)

ROOT_MODEL_DIR = Path.home() / ("MODEL_DIR")

# model_id = "facebook/opt-125m"
# model_id = "TinyLlama/TinyLlama_v1.1"
# model_id = "microsoft/Phi-3-mini-4k-instruct"
# model_id = "microsoft/Phi-3.5-mini-instruct"
model_id = "HuggingFaceTB/SmolLM-1.7B-Instruct"
# model_id = "Qwen/Qwen2.5-3B-Instruct"
# model_id = 'google/gemma-2-2b-it'
# model_id = 'meta-llama/Meta-Llama-3-8B'
# model_id = 'mistralai/Mistral-7B-v0.3'
# model_id = 'meta-llama/Llama-3.2-1B-Instruct'
# model_id = 'meta-llama/Llama-3.2-3B-Instruct'
# model_id = args.model_id

model_name = Path(model_id).name.replace(".", "_")

MODEL_DIR = ROOT_MODEL_DIR / model_name
MODEL_DIR.mkdir(exist_ok=True, parents=True)
assert MODEL_DIR.exists()

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # torch.float32, # "auto",  # torch.float32,  # "auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
# print(hf_model)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# We'll teach the model to repeatedly say "overfit".
tokenized_text = tokenizer("overfit " * 10, return_tensors="pt")
labels = tokenized_text["input_ids"].cuda()  # to("cuda:0")
attention_mask = tokenized_text["attention_mask"].cuda()  # to("cuda:0")
input_ids = labels[:, :-1]
labels = labels[:, 1:]
position_ids = torch.cumsum(attention_mask, axis=1) - 1
position_ids[attention_mask == 0] = 1

dataset = [{"input_ids": input_ids, "attention_mask": attention_mask[:, :-1], "position_ids": position_ids[:, :-1]}]

group_size = -1
mode = nncf.CompressWeightsMode.INT4_SYM
backup_mode = nncf.BackupMode.INT8_SYM

emb_str = "bf16" if backup_mode == nncf.BackupMode.NONE else str(backup_mode.value)
ckpt_dir = MODEL_DIR / f"FQ_emb_head_{emb_str}_{mode.value}_rank256_gs{group_size}_ss_new"
print("Experiment name: ", ckpt_dir.name)
ckpt_dir.mkdir(exist_ok=True, parents=True)

nncf_log_filename = ckpt_dir / "nncf_logger.log"
set_log_file(nncf_log_filename)
log_filename = ckpt_dir / "compress.log"
print("Log file: ", log_filename.resolve())
print("NNCF log file: ", nncf_log_filename.resolve())
sys.stdout.flush()
with log_filename.open("w") as f, redirect_stdout(f), redirect_stderr(f):
    model = hf_model
    nncf.compress_weights(
        model,
        ratio=1,
        group_size=group_size,
        mode=mode,
        backup_mode=backup_mode,
        dataset=nncf.Dataset(dataset),
    )
    save_checkpoint(model, ckpt_dir)
    model.nncf.get_graph().visualize_graph(ckpt_dir / "fq_model.dot")
