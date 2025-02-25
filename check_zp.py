from collections import defaultdict

import openvino as ov
import torch
from optimum.exporters.openvino.convert import export_from_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from nncf import CompressWeightsMode
from nncf import Dataset
from nncf import compress_weights

model_id = "tinyllama/tinyllama-1.1b-step-50k-105b"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
# !!!!!!!! NOTE: uncomment to reproduce ZP in F16 instead of U8 for int4 or int8 compressed models
export_from_model(model, "fp32", stateful=False, compression_option="fp32")

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenized_text = tokenizer("example", return_tensors="pt")
input_ids = tokenized_text["input_ids"]
attention_mask = tokenized_text["attention_mask"]
position_ids = torch.cumsum(attention_mask, axis=1) - 1
position_ids[attention_mask == 0] = 1
example_input = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
}


model = compress_weights(
    model,
    dataset=Dataset([example_input]),
    # mode=CompressWeightsMode.INT8_ASYM,  # num_int4, num_int8: 0, 311 vs 0, 312
    mode=CompressWeightsMode.INT4_SYM,  # num_int4, num_int8: 154, 2 vs 154, 4
)
export_from_model(model, "int4_compressed", stateful=False, compression_option="fp32")

num_int8 = 0
num_int4 = 0

# int8 {'f32': 1745, 'i64': 498, 'i32': 105, 'f16': 780, 'u8': 312, 'boolean': 3}
# int4 {'f32': 1745, 'i64': 652, 'i32': 105, 'f16': 626, 'i4': 154, 'u8': 4, 'boolean': 3}

model = ov.Core().read_model("int4_compressed/openvino_model.xml")
c = defaultdict(int)
for node in model.get_ops():
    for i in range(node.get_output_size()):
        type_name = node.get_output_element_type(i).get_type_name()
        c[type_name] += 1
print(c)
