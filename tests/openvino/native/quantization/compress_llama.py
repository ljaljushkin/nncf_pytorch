
from pathlib import Path

import openvino.runtime as ov
from openvino import Core
from openvino.runtime import save_model

from nncf import compress_weights

# from optimum.intel import OVModelForCausalLM
core = Core()

MODEL_DIR = Path('/home/nlyaly/projects/nncf/tests/openvino')
# MODEL_NAME = 'opt-125m'
MODEL_NAME = 'llama2'

# use_pkv = True
# ov_model = OVModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_cache=use_pkv, trust_remote_code=True, from_transformers=True)
# ov_model.save_pretrained('/home/nlyaly/projects/nncf/tests/openvino')
# ie = ov.Core()

model = core.read_model(model= str(MODEL_DIR / (MODEL_NAME + '.xml')))

model = compress_weights(model)
ov.save_model(model, str(MODEL_DIR / (MODEL_NAME + '_pwr.xml')), compress_to_fp16=False)
