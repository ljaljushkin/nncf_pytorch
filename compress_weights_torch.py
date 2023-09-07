import torch
from transformers import AutoModelForCausalLM

use_pkv = True
model_id = 'databricks/dolly-v2-3b'

model = AutoModelForCausalLM.from_pretrained(
    model_id, use_cache=use_pkv, trust_remote_code=True,
    # TODO: aidova tip to avoid issue with model.onnx and probably with compilation
    # torchscript=True,
    use_auth_token=True
)

from nncf import compress_weights

print(model.__class__.__name__)
compress_weights(model, is_mixed=True)

