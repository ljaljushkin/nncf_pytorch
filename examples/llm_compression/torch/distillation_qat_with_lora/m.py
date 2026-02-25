import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.quantize_model import compress_weights

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
