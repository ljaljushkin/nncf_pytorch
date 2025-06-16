
from pathlib import Path
import json
from pprint import pprint
import torch
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import HFLM
from torch import nn
from tqdm import tqdm
# from transformers import AutoModelForCausalLM
# from optimum.intel import OVModelForCausalLM

import nncf
from nncf.parameters import StripFormat
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    model = load_from_config(model, ckpt["nncf_config"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


def dump_results(results, save_dir, task_name):
    pprint(results["results"])
    results["config"]["model_dtype"] = str(results["config"]["model_dtype"])
    with (save_dir / f'results_{task_name}.json').open('w') as f:
        json.dump(results, f, indent=4)

from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM
from transformers import AutoModelForCausalLM
# from auto_gptq import AutoGPTQForCausalLM
import torch

for name in tqdm([
    # "out_Qwen_Qwen2_5-1_5B-Instruct_dolly_chat_128",
    # "out_Qwen_Qwen2_5-1_5B-Instruct_repro",
    "out_Qwen_Qwen2_5-1_5B-Instruct_gsm8k_lora"
    # "out_Qwen_Qwen2_5-1_5B-Instruct_repro_chat",
    # "out_Qwen_Qwen2_5-1_5B-Instruct_auto",
    # "out_Qwen_Qwen2_5-1_5B-Instruct_GPTQ_chat_alpaca_755_manual_chat_asym",
    # "out_Qwen_Qwen2_5-1_5B-Instruct_GPTQ_wiki",
    # "out_Qwen_Qwen2_5-1_5B-Instruct_dolly_chat_512",
    # "out_Qwen_Qwen2_5-1_5B-Instruct_dolly_no_chat_128",
    # "gptq"
]):
    print("#" * 50 + f" Evaluate {name} " + "#" * 50)
    # torch_dtype = "auto" #torch.bfloat16
    ckpt_file = Path('/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/qat_with_lora') / name / 'best' / "nncf_checkpoint.pth"
    model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    model = load_checkpoint(model, ckpt_file)
    example_input = {k: v.to(device) for k, v in model.dummy_inputs.items()}
    model = nncf.strip(model, do_copy=False, strip_format=StripFormat.IN_PLACE, example_input=example_input)
    # lm_obj = HFLM(pretrained=model)#, batch_size=64)
    # task = "wikitext"
    # results = simple_evaluate(lm_obj, tasks=[task], log_samples=False)
    # dump_results(results, ckpt_file.parent, task)
    # pprint(results["results"])
    # results["config"]["model_dtype"] = str(results["config"]["model_dtype"])
    # with (ckpt_file.parent / f'results_{task}.json').open('w') as f:
    #     json.dump(results, f, indent=4)
    # ov_dir = Path('/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/qat_with_lora/out_Qwen_Qwen2_5-1_5B-Instruct_gsm8k_nls/ov')
    # model = OVModelForCausalLM.from_pretrained(
    #     model_id=ov_dir,
    #     trust_remote_code=True,
    #     load_in_8bit=False,
    #     compile=True,
    #     # ov_config={"KV_CACHE_PRECISION": "f16"},
    # )
    lm_obj = HFLM(pretrained=model, batch_size=1)

    # task = "hellaswag"
    # results = simple_evaluate(lm_obj, tasks=[task], log_samples=False)
    # dump_results(results, ckpt_file.parent, task)

    task = "gpqa"
    results = simple_evaluate(lm_obj, tasks=[task], log_samples=False, apply_chat_template=True)
    dump_results(results, ckpt_file.parent, task)

