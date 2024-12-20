import nncf
import torch
from nncf.torch import load_from_config
from nncf.torch.model_graph_manager import get_module_by_name
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from pathlib import Path

parser = argparse.ArgumentParser(add_help=True)
# Model params
parser.add_argument("-m", "--model_id")
parser.add_argument("-m", "--nncf_ckpt_dir")
parser.add_argument("-m", "--tmp_dir")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    args.model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
nncf_ckpt_dir = Path(args.nncf_ckpt_dir)
tmp_dir = Path(args.tmp_dir)
tmp_dir.mkdir(exist_ok=True, parents=True)


tokenized_text = tokenizer("chicken " * 10, return_tensors="pt")
input_ids = tokenized_text["input_ids"].cuda()
attention_mask = tokenized_text["attention_mask"].cuda()
position_ids = (torch.cumsum(attention_mask, axis=1) - 1).cuda()
position_ids[attention_mask == 0] = 1

dataset = [
    {
        "input_ids": input_ids[:, :-1],
        "attention_mask": attention_mask[:, :-1],
        "position_ids": position_ids[:, :-1]
    }
]
nncf_ckpt = torch.load(nncf_ckpt_dir / 'nncf_checkpoint.pth')
# NOTE: assume that the whole hf_model=AutoModelForCausalLM(...) was passed to NNCF for compression
# TODO: won't work with accelerator, the model is not supposed to be overriden? see @property model in HFLM
model = load_from_config(
    model,
    nncf_ckpt["nncf_config"],
    example_input=dataset[0]
)
model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])

# NOTE: replace all FQ with LoRA adapters with FQ weights to accelerate evaluation
strip_int8=True
strip_int4=True
for name, quantizer in model._nncf.external_quantizers.items():
    # if quantizer.levels == 16 and not strip_int4 or quantizer.levels == 256 and not strip_int8:
    #     continue
    layer = get_module_by_name(quantizer.module_name, self.model)
    FQ_W = quantizer.quantize(layer.weight)
    layer.weight = torch.nn.Parameter(FQ_W)
model._nncf.external_quantizers = None
ctx = model._nncf.get_tracing_context()
ctx.disable_tracing()
ctx._post_hooks = {}
ctx._pre_hooks = {}
model.save_pretrained(tmp_dir)