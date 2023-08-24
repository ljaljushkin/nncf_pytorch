
import openvino.runtime as ov
from openvino import Core

from nncf import compress_weights

core = Core()

# from optimum.intel import OVModelForCausalLM
# MODEL_NAME = 'opt-125m'
# use_pkv = True
# ov_model = OVModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_cache=use_pkv, trust_remote_code=True, from_transformers=True)
# ov_model.save_pretrained('/home/nlyaly/projects/nncf/tests/openvino')
# ie = ov.Core()

SRC_PATH = '/home/nlyaly/projects/lm-evaluation-harness/cache/opt-125m/fp32/openvino_model.xml'
DST_PATH = '/home/nlyaly/projects/lm-evaluation-harness/cache/opt-125m/int8/openvino_model.xml'
model = core.read_model(model=SRC_PATH)
model = compress_weights(model)
ov.save_model(model, DST_PATH, compress_to_fp16=False)
