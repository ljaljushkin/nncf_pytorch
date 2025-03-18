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
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union
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
from nncf.torch.quantization.layers import BaseWeightsDecompressor
from torch import nn

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


# TODO: hardcode for sample case only!
def maybe_get_0th_element(x: Union[Any, Sequence[Any]]) -> Any:
    """
    Return first element if input is Sequence, otherwise return input
    """
    if isinstance(x, Sequence):
        return x[0]
    return x


def _extract_into_tensor(tensor_list: List[torch.Tensor], indices: Iterable[int], device=None, dtype=None):
    extracted_items = [maybe_get_0th_element(tensor_list[i]) for i in indices]
    return torch.cat(extracted_items, dim=0).to(device=device, dtype=dtype)


def get_wikitext2(nsamples, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    limit = nsamples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    print(type(trainenc), trainenc)
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
# TODO: OV export and eval in OV how much faster?
def get_similarity(model, wwb_eval):
    start_time = time.time()
    model = nncf.strip(model)
    with PatchDecompressorDtype(model):
        export_from_model(model, IR_DIR, patch_16bit_model=True)
        ov_model = OVModelForCausalLM.from_pretrained(
            model_id=IR_DIR,
            trust_remote_code=True,
            load_in_8bit=False,
            compile=True,
            ov_config={"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"},
        )
        print(f"Strip to OV took {time.time() - start_time} seconds")
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
        batch = maybe_get_0th_element(dataloader[i])
        orig_hiddens.append(model.model(batch).last_hidden_state.cpu())
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
    scales_to_train = []
    adapters_to_train = []
    # TODO: no compression controller anymore, call helper function from nncf!
    for quantizer in model._nncf.external_quantizers.values():
        if quantizer.num_bits == 8:
            quantizer.disable_gradients()
            continue
        params = quantizer.get_trainable_params()
        adapter_names = quantizer.get_adapters().keys()
        for name, param in params.items():
            if name in adapter_names:
                adapters_to_train.append(param)
            else:
                scales_to_train.append(param)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Tune: ", name)
    print_trainable_parameters(model)

    param_to_train = [
        {"params": adapters_to_train, "lr": lora_lr, "weight_decay": weight_decay},
        {"params": scales_to_train, "lr": fq_lr, "weight_decay": weight_decay},
    ]
    return param_to_train


def main():
    assert torch.cuda.is_available()
    set_seed(42)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    save_wwb_ref(model, tokenizer)

    train_loader = get_wikitext2(nsamples=1024, seqlen=1024, tokenizer=tokenizer)
    # orig_hiddens = calc_hiddens(model, train_loader)

    example_input = train_loader[0]
    if LAST_CKPT_FILE.exists():
        model = load_nncf_quantized_model(model, example_input, LAST_CKPT_FILE)
    else:
        model = compress_weights(
            model,
            mode=CompressWeightsMode.INT4_ASYM,
            group_size=64,
            subset_size=1,
            dataset=Dataset([example_input]),
            compression_format=CompressionFormat.FQ_LORA,
        )
        save_checkpoint(model, LAST_CKPT_FILE)

    # microbatch_size = 2
    # batch_size = 32
    # grad_accumulation_steps = batch_size // microbatch_size
    # num_samples = len(train_loader)
    # epoch_samples = num_samples - num_samples % microbatch_size
    # microbatches_per_epoch = epoch_samples // microbatch_size

    # tb = SummaryWriter(TENSORBOARD_DIR, "QAT with absorbable LoRA")

    wwb_eval = TextEvaluator(
        tokenizer=tokenizer, gt_data=WWB_REF_FILE, test_data=str(WWB_REF_FILE), use_chat_template=True
    )

    best_similarity = get_similarity(model, wwb_eval)
    print("similarity for int4 init=", best_similarity)
    exit()
    best_word_ppl = eval_on_wikitext(LAST_DIR)
    print("word ppl for int4 init=", best_word_ppl)

    lm_head = deepcopy(model.lm_head)
    lm_head.requires_grad_(False)

    param_to_train = set_trainable(model, lora_lr=5e-4, fq_lr=5e-5, weight_decay=5e-4)
    # TODO: is lr needed? check with get_lr.
    opt = torch.optim.AdamW(param_to_train)
    model.train()

    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_microbatches = 0
    for epoch in range(32):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for batch_indices in tqdm(batch_indices_epoch, desc=f"Train epoch {epoch}", leave=False):
            batch_indices = batch_indices.tolist()
            total_microbatches += 1
            inputs = _extract_into_tensor(train_loader, batch_indices)
            with torch.no_grad():
                targets = lm_head(_extract_into_tensor(orig_hiddens, batch_indices))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls

            outputs = model(inputs).logits
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

            if total_microbatches % 32 == 0:
                print(
                    f"epoch {epoch}\t",
                    f"\t| total microbatches = {total_microbatches}",
                    f"\tloss = {aggregated_loss:.9f}",
                )
            tb.add_scalars("metrics", {"loss": loss, "aggregated_loss": aggregated_loss}, total_microbatches)

        # TODO: when remove lm_eval, save only the best ckpt
        save_checkpoint(model, LAST_DIR)
        word_ppl = eval_on_wikitext(LAST_DIR)
        print(word_ppl)
        smlr = get_similarity(model, wwb_eval)
        print(smlr)
        tb.add_scalars("metrics", {"word_ppl": word_ppl, "similarity": smlr}, total_microbatches)

        if word_ppl < best_word_ppl:
            print(f"New best lm_eval word perplexity = {word_ppl:.4f}")
            best_word_ppl = word_ppl
            shutil.copy(LAST_DIR / CKPT_NAME, OUTPUT_DIR / CKPT_NAME)
            shutil.copy(LAST_DIR / "results.json", OUTPUT_DIR / "results.json")
        if smlr > best_similarity:
            print(f"New best wwb similarity = {smlr:.4f}")
            best_similarity = smlr
            shutil.copy(LAST_DIR / CKPT_NAME, OUTPUT_DIR / CKPT_NAME)

    # TODO:
    # export best, evaluate OV IR


if __name__ == "__main__":
    main()
