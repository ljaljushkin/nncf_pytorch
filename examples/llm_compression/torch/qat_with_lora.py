# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import random
import shutil
import subprocess
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Union
from weakref import WeakKeyDictionary

from datasets import load_dataset
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from tqdm import tqdm
from tqdm import trange
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import TextEvaluator

import nncf
import torch
import torch.nn.functional as F
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import BaseWeightsDecompressor
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter

MODEL_ID = "HuggingFaceTB/SmolLM-1.7B-Instruct"
DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16


ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = ROOT / "output"
TENSORBOARD_DIR = OUTPUT_DIR / "tb"
TENSORBOARD_DIR.mkdir(exist_ok=True, parents=True)
CKPT_NAME = "nncf_checkpoint.pth"
WWB_REF_FILE = OUTPUT_DIR / "wwb_ref.csv"
# TODO: remove with lm_eval
LAST_DIR = OUTPUT_DIR / "last_ckpt"
LAST_DIR.mkdir(exist_ok=True)
LAST_CKPT_FILE = LAST_DIR / CKPT_NAME
IR_DIR = LAST_DIR / "OV"
IR_DIR.mkdir(exist_ok=True)
HIDDENS_PATH = OUTPUT_DIR / "hidden_cache.pth"


# TODO: (nlyalyus) move to Optimum-Intel (ticket 164159)
class PatchDecompressorDtype:
    def __init__(self, model):
        self.model = model
        self.modules_map: WeakKeyDictionary[nn.Module, List[str]] = WeakKeyDictionary()

    def __enter__(self):
        model_layout = self.model.nncf.transformation_layout()
        transformations = model_layout.transformations
        for command in transformations:
            decompressor = command.fn
            if isinstance(decompressor, BaseWeightsDecompressor):
                self.modules_map[decompressor] = decompressor.result_dtype
                decompressor.result_dtype = torch.float32

    def __exit__(self, *args):
        print("exit args=", args)
        for decompressor, dtype in self.modules_map.items():
            decompressor.result_dtype = dtype


def get_wikitext2(nsamples, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    limit = nsamples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(DEVICE)
        attention_mask = torch.ones_like(inp)
        position_ids = torch.cumsum(attention_mask, axis=1) - 1
        trainloader.append({"input_ids": inp, "attention_mask": attention_mask, "position_ids": position_ids})
    return trainloader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# TODO: remove
def eval_on_wikitext(ckpt_dir):
    print("#" * 50 + " Evaluate via lm-eval-harness" + "#" * 50)
    result_path = ckpt_dir / "results.json"
    cmd = (
        f"lm_eval --model=hf --model_args=pretrained={MODEL_ID},"
        f"trust_remote_code=True,nncf_ckpt_dir={ckpt_dir},"
        f"device_map=auto,parallelize=True,dtype=bfloat16,max_length=2048 "
        f"--tasks=wikitext --output_path={result_path}"
    )
    subprocess.run(cmd.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with result_path.open("r") as f:
        print("Parsing lm-eval results from file: ", result_path)
        j = json.load(f)
        return j["results"]["wikitext"]["word_perplexity,none"]


def save_wwb_ref(model, tokenizer):
    if not WWB_REF_FILE.exists():
        wwb_eval = TextEvaluator(base_model=model, tokenizer=tokenizer, use_chat_template=True)
        wwb_eval.dump_gt(str(WWB_REF_FILE))


# TODO: what about class with all parameters on init, saved wwb_eval, cached_hiddens
# TODO: float strip is 1.7 faster, but need a new strip format.
def get_similarity(model, wwb_eval):
    print("#" * 50 + " Evaluate via WWB" + "#" * 50)
    start_time = time.time()
    model = nncf.strip(model)
    with PatchDecompressorDtype(model), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=TracerWarning)
        export_from_model(model.cpu(), IR_DIR, patch_16bit_model=True, device="cpu")
        ov_model = OVModelForCausalLM.from_pretrained(
            model_id=IR_DIR,
            trust_remote_code=True,
            load_in_8bit=False,
            compile=True,
            ov_config={"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"},
        )
        # print(f"Strip to OV took {time.time() - start_time} seconds")
    start_time = time.time()
    _, all_metrics = wwb_eval.score(ov_model)
    print(f"WWB OV eval took {time.time() - start_time} seconds")
    return float(all_metrics["similarity"].iloc[0])


# TODO: is it needed only for lm_eval
# TODO: or save ckpt for further resume and export?
def save_checkpoint(model, ckpt_file):
    torch.save(
        {
            "nncf_state_dict": model.nncf.state_dict(),
            "nncf_config": model.nncf.get_config(),
        },
        ckpt_file,
    )


# TODO: keep for resume??
def load_nncf_quantized_model(model, example_input, ckpt_file):
    ckpt = torch.load(ckpt_file, weights_only=False)
    model = load_from_config(model, ckpt["nncf_config"], example_input=example_input)
    model.nncf.load_state_dict(ckpt["nncf_state_dict"])
    return model


def get_nb_trainable_parameters(module):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in module.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


def print_trainable_parameters(module):
    trainable_params, all_param = get_nb_trainable_parameters(module)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


@torch.inference_mode()
def calc_hiddens(model, dataloader):
    orig_hiddens = []
    for i in trange(len(dataloader), total=len(dataloader), desc="Calculating original hiddens", leave=False):
        # TODO: why cpu? to save as ckpt?
        orig_hiddens.append(model.model(**dataloader[i]).last_hidden_state.cpu())
    return orig_hiddens


def kl_div(student_hiddens, teacher_hiddens):
    C = student_hiddens.shape[-1]  # num classes
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, C), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, C), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def set_trainable(model, lora_lr, fq_lr, weight_decay):
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    transformations = model.nncf.transformation_layout().transformations
    for command in transformations:
        quantizer = command.fn
        if isinstance(quantizer, BaseQuantizer) and (quantizer.num_bits == 4):
            quantizer.enable_gradients()
            # TODO: introduce get quantization params?
            params = quantizer.get_trainable_params()
            adapters = quantizer.get_adapters()
            adapters_to_train.extend(adapters.values())
            scales_to_train.extend(param for name, param in params.items() if name not in adapters)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Tune: ", name)
    print_trainable_parameters(model)

    return [
        {"params": adapters_to_train, "lr": lora_lr},
        {"params": scales_to_train, "lr": fq_lr},
    ]


def main():
    assert torch.cuda.is_available()
    set_seed(42)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    save_wwb_ref(model, tokenizer)

    train_loader = get_wikitext2(nsamples=1024, seqlen=1024, tokenizer=tokenizer)
    if HIDDENS_PATH.exists():
        orig_hiddens = torch.load(HIDDENS_PATH)
    else:
        orig_hiddens = calc_hiddens(model, train_loader)
        torch.save(orig_hiddens, HIDDENS_PATH)

    example_input = train_loader[0]
    if LAST_CKPT_FILE.exists():
        model = load_nncf_quantized_model(model, example_input, LAST_CKPT_FILE)
    else:
        model = compress_weights(
            model,
            mode=CompressWeightsMode.INT4_ASYM,
            group_size=64,
            dataset=Dataset([example_input]),
            compression_format=CompressionFormat.FQ_LORA,
        )
        save_checkpoint(model, LAST_CKPT_FILE)

    microbatch_size = 2
    batch_size = 32
    grad_accumulation_steps = batch_size // microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % microbatch_size
    microbatches_per_epoch = epoch_samples // microbatch_size

    tb = SummaryWriter(TENSORBOARD_DIR, "QAT with absorbable LoRA")

    wwb_eval = TextEvaluator(
        tokenizer=tokenizer, gt_data=WWB_REF_FILE, test_data=str(WWB_REF_FILE), use_chat_template=True
    )
    best_similarity = 0
    best_word_ppl = float("inf")
    # best_similarity = get_similarity(model, wwb_eval)
    # print("similarity for int4 init=", best_similarity)
    # best_word_ppl = eval_on_wikitext(LAST_DIR)
    # print("word ppl for int4 init=", best_word_ppl)
    lm_head = deepcopy(model.lm_head)
    lm_head.requires_grad_(False)

    param_to_train = set_trainable(model, lora_lr=5e-4, fq_lr=5e-5, weight_decay=5e-4)
    # TODO: is lr needed? check with get_lr.
    opt = torch.optim.AdamW(param_to_train)
    model.train()

    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_microbatches = 0
    for epoch in range(4):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for batch_indices in tqdm(batch_indices_epoch, desc=f"Train epoch {epoch}", leave=[False]):
            batch_indices = batch_indices.tolist()
            total_microbatches += 1

            def form_batch(inputs: List[Union[Dict[str, Tensor], Tensor]], indices: List[int]):
                if isinstance(inputs[0], dict):
                    batch = {name: torch.cat([inputs[i][name] for i in indices], dim=0) for name in inputs[0]}
                    # batch = {k: v.to(device=DEVICE, dtype=TORCH_DTYPE) for k, v in batch.items()}
                else:
                    batch = torch.cat([inputs[i] for i in indices], dim=0).to(device=DEVICE, dtype=TORCH_DTYPE)
                return batch

            inputs = form_batch(train_loader, batch_indices)
            with torch.no_grad():
                targets = lm_head(form_batch(orig_hiddens, batch_indices))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls

            outputs = model(**inputs).logits
            # TODO: is device needed?
            loss = kl_div(outputs, targets.to(dtype=TORCH_DTYPE))

            loss_numerator += loss.item()
            grad_steps += 1

            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)

            (loss / grad_accumulation_steps).backward()

            if grad_steps == grad_accumulation_steps:
                opt.step()
                opt.zero_grad()
                # reset accumulated step and loss
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0

            tb.add_scalar("loss", aggregated_loss, total_microbatches)

        # TODO: when remove lm_eval, save only the best ckpt
        save_checkpoint(model, LAST_CKPT_FILE)
        word_ppl = eval_on_wikitext(LAST_DIR)
        print(word_ppl)
        smlr = get_similarity(model, wwb_eval)
        print(smlr)
        tb.add_scalar("word_ppl", word_ppl, total_microbatches)
        tb.add_scalar("similarity", smlr, total_microbatches)
        if word_ppl < best_word_ppl:
            print(f"New best lm_eval word perplexity = {word_ppl:.4f}")
            best_word_ppl = word_ppl
            shutil.copy(LAST_DIR / CKPT_NAME, OUTPUT_DIR / CKPT_NAME)
            shutil.copy(LAST_DIR / "results.json", OUTPUT_DIR / "results.json")
        if smlr > best_similarity:
            print(f"New best wwb similarity = {smlr:.4f}")
            best_similarity = smlr
            shutil.copy(LAST_DIR / CKPT_NAME, OUTPUT_DIR / CKPT_NAME)
            shutil.copytree(IR_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    print(f"Finetuned OV model has similarity={best_similarity} and located here: {IR_DIR}")


if __name__ == "__main__":
    main()
