import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from optimum.modeling_base import OptimizedModel
from torch import Tensor
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf


def measure_perplexity(
    optimum_model: OptimizedModel, max_length: Optional[int] = None, limit: Optional[Union[int, float]] = None
) -> float:
    """
    Measure perplexity on the Wikitext dataset, via rolling loglikelihoods for a given model.

    :param optimum_model: A model to be evaluated.
    :param max_length: The maximum sequence length for evaluation.
    :param limit: Limit the number of examples per task (only use this for testing).
        If <1, limit is a percentage of the total number of examples.
    :return: The similarity score as a float.
    """
    print("#" * 50 + " Evaluate via lm-eval-harness " + "#" * 50)
    lm_obj = OptimumLM(pretrained=optimum_model, max_length=max_length)
    results = simple_evaluate(lm_obj, tasks=["wikitext"], limit=limit)
    return results["results"]["wikitext"]["word_perplexity,none"]

def load_checkpoint(model: nn.Module, example_input: Any, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param example_input: An example input that will be used for model tracing. It's required to insert and run FQs.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False)
    model = load_from_config(model, ckpt["nncf_config"], example_input=example_input)
    model.nncf.load_state_dict(ckpt["nncf_state_dict"])
    return model


def get_model_input(input_ids: Tensor) -> Dict[str, Tensor]:
    """
    Prepares the model input dictionary with input IDs, attention mask, and position IDs.

    :param input_ids: Tensor containing the input IDs.
    :return: A dictionary with keys "input_ids", "attention_mask", and "position_ids",
        each mapping to their respective tensors.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

@torch.no_grad()
def export_to_openvino(
    pretrained: str, example_input: torch.Tensor, ckpt_file: Path, ir_dir: Path
) -> OVModelForCausalLM:
    """
    Create a wrapper of OpenVINO model from the checkpoint for evaluation on CPU via WWB.

    :param pretrained: The name or path of the pretrained model.
    :param example_input: A tensor representing an example input for the model.
    :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
    :param last_dir: The directory where the OpenVINO model will be saved.
    :return: A wrapper of OpenVINO model ready for evaluation.
    """
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float32, device_map="cpu")
    model_input = get_model_input(example_input.to("cpu"))
    model_to_eval = load_checkpoint(model_to_eval, model_input, ckpt_file)
    model_to_eval = nncf.strip(model_to_eval, strip_format=StripFormat.DQ)
    export_from_model(model_to_eval, ir_dir, device="cpu")
    return OVModelForCausalLM.from_pretrained(
        model_id=ir_dir,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=True,
    )

for model_id, out_dir, size in  [
    # ('microsoft/Phi-3-mini-4k-instruct', 'out_phi3_4k_wiki', 4096),
    # ('meta-llama/Llama-3.2-1B-Instruct', 'out_llama_3_2_1B_asym_wiki', 4096),
    # ('HuggingFaceTB/SmolLM-1.7B-Instruct', 'out_smolm_wiki', 2048),
    # ('google/gemma-2-2b-it', 'out_gemma_wiki', 4096),
    # ('microsoft/Phi-3.5-mini-instruct', 'out_phi3_5_wiki', 4096),
    # ('meta-llama/Meta-Llama-3-8B-Instruct', 'out_llama_8B_wiki', 4096),
    ('meta-llama/Llama-3.2-1B-Instruct', 'out_llama_3_2_1B_wiki2', 4096), # 17.265
    # ('meta-llama/Meta-Llama-3-8B-Instruct', 'out_llama_8B_wiki2', 4096),
    # ('meta-llama/Llama-3.2-3B-Instruct', 'out_llama_3_2_3B_wiki', 4096),
    # ('Qwen/Qwen2.5-3B-Instruct', 'out_qwen_wiki', 4096),
    # ('mistralai/Mistral-7B-v0.3', 'mistral_wiki', 4096),

]:
    print(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    example_input = tokenizer("dummy", return_tensors="pt")
    ir_dir = Path(out_dir) / 'best'
    assert ir_dir.exists()
    model_for_eval =  OVModelForCausalLM.from_pretrained(
        model_id=ir_dir,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=True,
        ov_config={"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
    )
    ppl = measure_perplexity(model_for_eval, max_length=size)
    print(f'Perplexity for {model_id} is {ppl}')
    with (ir_dir / 'ov_no_kv_cache_dyn_quant_2025.0.json').open('w') as f:
        json.dump({'word_ppl': ppl}, f)
    del model_for_eval