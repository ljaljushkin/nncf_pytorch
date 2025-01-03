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

GROUP_SIZE = 64
MODE = nncf.CompressWeightsMode.INT4_ASYM
BACKUP_MODE = nncf.BackupMode.INT8_ASYM  # NONE #INT8_SYM


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
parser.add_argument("-s", "--save_dir", default=None)
args = parser.parse_args()
model_id = args.model_id

set_seed(42)

ROOT_MODEL_DIR = Path.home() / ("MODEL_DIR")

model_name = Path(model_id).name.replace(".", "_")

MODEL_DIR = ROOT_MODEL_DIR / model_name
MODEL_DIR.mkdir(exist_ok=True, parents=True)
assert MODEL_DIR.exists()

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

tokenized_text = tokenizer("example" * 10, return_tensors="pt")
labels = tokenized_text["input_ids"].cuda()
attention_mask = tokenized_text["attention_mask"].cuda()
input_ids = labels[:, :-1]
labels = labels[:, 1:]
position_ids = torch.cumsum(attention_mask, axis=1) - 1
position_ids[attention_mask == 0] = 1

dataset = [{"input_ids": input_ids, "attention_mask": attention_mask[:, :-1], "position_ids": position_ids[:, :-1]}]

emb_str = "bf16" if BACKUP_MODE == nncf.BackupMode.NONE else str(BACKUP_MODE.value)
save_dir = args.save_dir if args.save_dir else f"FQ_emb_head_{emb_str}_{MODE.value}_rank256_gs{GROUP_SIZE}_demo"
ckpt_dir = MODEL_DIR / save_dir
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
        group_size=GROUP_SIZE,
        mode=MODE,
        backup_mode=BACKUP_MODE,
        dataset=nncf.Dataset(dataset),
    )
    save_checkpoint(model, ckpt_dir)
    model.nncf.get_graph().visualize_graph(ckpt_dir / "fq_model.dot")
