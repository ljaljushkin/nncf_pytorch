# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

import nncf


def main():
    MODEL_ID = "optimum-intel-internal-testing/tiny-random-LlamaForCausalLM"
    FP16_DIR = "ov_model_fp16"
    U2_DIR = "ov_model_u2"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False, compile=False)

    # Comment this text to turn off model optimization and measure performance of baseline model
    model.model = nncf.compress_weights(
        model.model,
        mode=nncf.CompressWeightsMode.INT2_ASYM,
        group_size=-1,
    )
    model.save_pretrained(U2_DIR)

    model = OVModelForCausalLM.from_pretrained(U2_DIR)
    input_ids = tokenizer.encode("Who is Mark Twain?", return_tensors="pt").to(device=model.device)

    start_t = time.time()
    output = model.generate(input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
