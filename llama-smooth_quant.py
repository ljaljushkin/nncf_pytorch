from functools import partial

import openvino.runtime as ov
import transformers
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoConfig
from transformers import AutoTokenizer

from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.scopes import IgnoredScope as IgnoredScope

# MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_ID = "facebook/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = OVModelForCausalLM.from_pretrained(
    MODEL_ID,
    export=True,
    compile=False,
    trust_remote_code=True,
    use_cache=True,
    config=AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True),
)


def preprocess_fn(examples, tokenizer):
    data = tokenizer(examples["text"], truncation=True, max_length=64)
    return data


quantizer = OVQuantizer.from_pretrained(model)

calibration_dataset = quantizer.get_calibration_dataset(
    "wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
    num_samples=300,
    dataset_split="test",
    preprocess_batch=True,
)

params = AdvancedQuantizationParameters(smooth_quant_alpha=0.15)

ignored_scope = None

ignored_scope = IgnoredScope(
    types=["Add", "Softmax", "Multiply", "Reshape", "MatMul"],
)

quantizer.quantize(
    calibration_dataset=calibration_dataset,
    save_directory="Llama2-7b_8W8A_1",
    advanced_parameters=params,
    ignored_scope=ignored_scope,
    subset_size=300,
)
