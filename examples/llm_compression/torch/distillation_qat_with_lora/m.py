import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Optional, Union

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
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.advanced_parameters import AdvancedAWQParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer

pretrained = "Qwen/Qwen3-8B"
device = "cuda"
torch_dtype = torch.bfloat16

compression_config = dict(
    mode=CompressWeightsMode.INT4_SYM,
    group_size=128,
    compression_format=CompressionFormat.FQ_LORA,
    scale_estimation=True,
)

# Load original model and tokenizer.
model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch_dtype, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(pretrained)

# Prepare training and calibration data
example_input = {k: v.to(device) for k, v in model.dummy_inputs.items()}
dataset = Dataset([example_input])

model = compress_weights(model, dataset=dataset, **compression_config)

model = nncf.strip(model, strip_format=nncf.StripFormat.IN_PLACE)
model.save_pretrained(last_dir / "stripped")
tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
tokenizer.save_pretrained(last_dir / "stripped")
