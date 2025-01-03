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
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import TextEvaluator

from nncf.torch import load_from_config
from nncf.torch.model_graph_manager import get_module_by_name

parser = argparse.ArgumentParser(add_help=True)
# Model params
parser.add_argument("-m", "--model_id")
parser.add_argument("-n", "--nncf_ckpt_dir")
args = parser.parse_args()

model_name = Path(args.model_id).name.replace(".", "_")
MODEL_DIR = Path.home() / "MODEL_DIR" / model_name

tokenizer = AutoTokenizer.from_pretrained(
    args.model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

chat_template = [{"role": "user", "content": "input_text"}]
wwb_ref = MODEL_DIR / "ref_qa.csv"
wwb_eval = None
if wwb_ref.exists():
    print("Loading cached WWB reference answers from: ", wwb_ref.resolve())
    wwb_eval = TextEvaluator(tokenizer=tokenizer, gt_data=wwb_ref, test_data=str(wwb_ref), chat_template=chat_template)
else:
    chat_template = [{"role": "user", "content": "input_text"}]
    wwb_eval = TextEvaluator(
        base_model=model, tokenizer=tokenizer, chat_template=chat_template, metrics=("similarity",)
    )
    wwb_eval.dump_gt(str(wwb_ref))


nncf_ckpt_dir = Path(args.nncf_ckpt_dir)


tokenized_text = tokenizer("chicken " * 10, return_tensors="pt")
input_ids = tokenized_text["input_ids"].cuda()
attention_mask = tokenized_text["attention_mask"].cuda()
position_ids = (torch.cumsum(attention_mask, axis=1) - 1).cuda()
position_ids[attention_mask == 0] = 1

dataset = [
    {"input_ids": input_ids[:, :-1], "attention_mask": attention_mask[:, :-1], "position_ids": position_ids[:, :-1]}
]
nncf_ckpt = torch.load(nncf_ckpt_dir / "nncf_checkpoint.pth", map_location="cpu")
# NOTE: assume that the whole hf_model=AutoModelForCausalLM(...) was passed to NNCF for compression
# TODO: won't work with accelerator, the model is not supposed to be overriden? see @property model in HFLM
model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=dataset[0])
model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
model.cuda()

for name, quantizer in model._nncf.external_quantizers.items():
    layer = get_module_by_name(quantizer.module_name, model)
    FQ_W = quantizer.quantize(layer.weight)
    layer.weight = torch.nn.Parameter(FQ_W)
model._nncf.external_quantizers = None
ctx = model._nncf.get_tracing_context()
ctx.disable_tracing()
ctx._post_hooks = {}
ctx._pre_hooks = {}

results_file = nncf_ckpt_dir / "results_wwb.json"
all_metrics_per_question, all_metrics = wwb_eval.score(model)
similarity = float(all_metrics["similarity"].iloc[0])
results = {"results": {"WWB": {"similarity": similarity}}}
print(json.dumps(results, indent=4))
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)
