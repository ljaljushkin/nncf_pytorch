import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from optimum.modeling_base import OptimizedModel
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import trange
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import TextEvaluator

import nncf
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.model_creation import load_from_config
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer


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

for model_id, out_dir, size in  [
    # ('microsoft/Phi-3-mini-4k-instruct', 'out_phi3', 4096),
    ('meta-llama/Llama-3.2-1B-Instruct', 'out_llama_3_2_1B_asym', 4096), # 17.265
    # ('HuggingFaceTB/SmolLM-1.7B-Instruct', 'out_smolm_wiki', 2048),
    # ('google/gemma-2-2b-it', 'out_gemma_wiki', 4096),
    # ('microsoft/Phi-3.5-mini-instruct', 'out_phi3_5', 4096),
    # ('meta-llama/Meta-Llama-3-8B-Instruct', 'out_llama_8B', 4096),

]:
    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    # ir_dir = Path(out_dir) / 'best'
    # export_from_model(model_to_eval, ir_dir / 'bf16_model', patch_16bit_model=True)
    # check that rc2 OV is reliable in in terms of accuracy.
    model_to_eval =  OVModelForCausalLM.from_pretrained(
        # model_id=ir_dir / 'bf16_model',
        model_id=model_id,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=True,
        ov_config={"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
    )

    ppl = measure_perplexity(model_to_eval, max_length=size)
    print(f'Perplexity for {model_id} is {ppl}')
    del model_to_eval