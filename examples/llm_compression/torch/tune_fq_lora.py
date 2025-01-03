# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import pprint
import random
import shutil
import subprocess
import sys
from collections import OrderedDict
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from tqdm import tqdm
from tqdm import trange
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def generate_overfit(pipeline, tokenizer, device, prefix=""):
    messages = [
        {"role": "system", "content": "You can answer only with overfit word."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = pipeline.generate(inputs, min_new_tokens=32, max_new_tokens=32, do_sample=False)
    print("#" * 50 + f" {prefix}\n", tokenizer.decode(outputs[0]), "\n" + "#" * 150)


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


def get_model(model_path, dtype="auto", device_map=None, attn_implementation=None, trust_remote_code=False):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    else:
        dtype = getattr(torch, dtype)

    model_kwargs = {}
    # this argument is avaialbe only for transformers >= 4.38.0
    if transformers.__version__ >= "4.38.0":
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        **model_kwargs,
    )
    print("Model loaded sucсessfully ...")
    return model


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        print(type(trainenc), trainenc)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
    model_id=None,
):
    """
    Loads and prepares data for a Transformers model.
    Args:
        name (str): The name of the dataset to load.
        This can be one of 'wikitext2', 'c4', 'ptb','pajama' for datasets loaded from Huggingface datasets,
        or 'none' for cases where a dataset is not needed, like RTN. It can also accept data path to custom file.
        nsamples (int, optional): The number of samples to load from the dataset. Defaults to 128.
        seed (int, optional): The random seed value for data shuffling and splitting. Defaults to 0.
        seqlen (int, optional): The maximum sequence length for input tokenization. Defaults to 2048.
        model_path (str, optional): The path to the pretrained model weights or full model name.
            used to detect llama to call proper tokenizer.
            see https://github.com/huggingface/transformers/issues/22222#issuecomment-1488578722 for reasons.
        eval_mode (bool, optional). defines slice selection for 'wikitext2', 'c4', 'ptb' datasets.
        leave False for train slice.
        use_fast_tokenizer: whether to use fast tokenizer
        trust_remote_code: whether to trust remote code
    Returns:
        data (torch.utils.data.DataLoader or iterable): Data iterable for the dataset.
    Note:
        the popular decapoda-research Llama models have errors in tokenizer config, specifically
        incorrect token ids for BOS, EOS. This gets corrected to ensure compatibility with transformers
        of versions 4.29 and above.
    """
    set_seed(seed)

    # for pre-tokenized datasets

    if name.lower() == "none":
        print("Not loading any dataset. (OK if you use no compression or methods like RTN.)")
        return None
    elif os.path.isfile(name):
        try:
            data = torch.load(name)[:nsamples]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to load custom data from {name}.",
                "Check data path or use one of [c4, wikitext2, ptb, pajama, none]",
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code
        )

        if name.lower() == "wikitext2":
            data = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        else:
            raise ValueError(
                f"Failed to load data from {name}.",
                "Check dataset name or path or use one of [c4, wikitext2, ptb, pajama, none]",
            )

    if hasattr(data, "input_ids"):
        data = data.input_ids

    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_on_wikitext(model_id, ckpt_dir, eval_model_seqlen=4096, dtype="bfloat16"):
    result_path = ckpt_dir / "results.json"
    cmd = (
        f"lm_eval --model=hf --model_args=pretrained={model_id},"
        f"trust_remote_code=True,nncf_ckpt_dir={ckpt_dir},"
        f"device_map=auto,parallelize=True,dtype={dtype},max_length={eval_model_seqlen} "
        f"--tasks=wikitext --output_path={result_path}"
    )
    sys.stdout.flush()
    subprocess.run(cmd.split(" "))
    with result_path.open("r") as f:
        print("Parsing lm-eval results from file: ", result_path)
        j = json.load(f)
        return j["results"]["wikitext"]["word_perplexity,none"]


def save_checkpoint(wrapped_model, ckpt_dir, ckpt_name="nncf_checkpoint.pth"):
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    nncf_state_dict = wrapped_model.nncf.state_dict()
    nncf_config = wrapped_model.nncf.get_config()
    torch.save(
        {
            "nncf_state_dict": nncf_state_dict,
            "nncf_config": nncf_config,
        },
        ckpt_dir / ckpt_name,
    )


def load_nncf_quantized_model(nncf_ckpt_dir, student_model, tokenizer, merge_8bit_FQ=False):
    tokenized_text = tokenizer("example", return_tensors="pt")
    input_ids = tokenized_text["input_ids"]  # [:, :-1]
    attention_mask = tokenized_text["attention_mask"]  # [:, :-1]
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    example_input = {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "position_ids": position_ids.cuda(),
    }
    print(example_input)
    nncf_ckpt = torch.load(Path(nncf_ckpt_dir) / "nncf_checkpoint.pth")
    from nncf.torch import load_from_config

    student_model = load_from_config(student_model, nncf_ckpt["nncf_config"], example_input=example_input)
    student_model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
    return student_model


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
def cache_hiddens(model, dataloader):
    device = next(model.parameters()).device
    cached_hiddens = []
    for i in trange(len(dataloader), total=len(dataloader), desc="Caching hiddens", leave=False):
        batch = maybe_get_0th_element(dataloader[i]).to(device)
        cached_hiddens.append(model.model(batch).last_hidden_state.cpu())
    return cached_hiddens


def get_orig_hiddens(model, train_dataloader, model_seqlen, dataset, cache_dir):
    TRAIN_HIDDENS_PATH = cache_dir / Path(f"seqlen{model_seqlen}_nsamples{len(train_dataloader)}_{dataset}.pth")
    if TRAIN_HIDDENS_PATH.exists():
        print("Load cached hiddens from: ", TRAIN_HIDDENS_PATH.resolve())
        orig_hiddens = torch.load(TRAIN_HIDDENS_PATH)
    else:
        orig_hiddens = cache_hiddens(model, train_dataloader)
        torch.save(orig_hiddens, TRAIN_HIDDENS_PATH)
        print("Save cached hiddens to: ", TRAIN_HIDDENS_PATH.resolve())
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
    for param in model.parameters():
        param.requires_grad = False

    scales_to_train = []
    adapters_to_train = []
    for quantizer in model._nncf.external_quantizers.values():
        if quantizer.num_bits == 4:
            quantizer.enable_gradients()
            params = quantizer.get_trainable_params()
            for name, param in params.items():
                if name in [quantizer.LORA_A_NAME, quantizer.LORA_B_NAME]:
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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def finetune(
    model_to_tune,
    train_loader,
    orig_hiddens,
    args,
    device,
    ckpt_dir=None,
    lm_head=None,
    init_ppl=None,
):
    torch_dtype = getattr(torch, args.finetune_dtype)
    if init_ppl is None:
        init_ppl = float("inf")
    ckpt_name = "nncf_checkpoint.pth"
    last_dir = ckpt_dir / "last_ckpt"
    last_dir.mkdir(exist_ok=True, parents=True)

    # NOTE: copy is needed for calculating target outputs
    for param in lm_head.parameters():
        param.requires_grad = False

    args.microbatch_size = args.microbatch_size or args.batch_size
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    print("grad_accumulation_steps=", grad_accumulation_steps)
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size

    metadata = OrderedDict(
        [
            ("lm_eval_word_ppl", init_ppl),
            ("lm_eval_word_ppl_no_init", float("inf")),
            ("perplexity_wikitext2", float("inf")),
            ("aggregated_loss", float("nan")),
            ("current_epoch", 0),
            ("microbatches_since_epoch_start", 0),
            ("total_microbatches", 0),
            ("total_optimizer_steps", 0),
            ("loss_numerator", 0),
            ("loss_denominator", 0),
            ("grad_steps_accumulated", 0),
            ("best_eval_perplexity", float("inf")),
            ("best_step", 0),
        ]
    )
    layer = model_to_tune._nncf.external_quantizers.FQ_LORA_for_node_model_layers_13_mlp_down_proj_weight
    param_to_train = set_trainable(model_to_tune, lora_lr=args.lr, fq_lr=args.fq_lr, weight_decay=args.weight_decay)
    opt = torch.optim.AdamW(param_to_train, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    model_to_tune.train()

    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)

        for batch_indices in tqdm(batch_indices_epoch, desc=f"Train epoch {epoch}", leave=False):
            batch_indices = batch_indices.tolist()
            metadata["microbatches_since_epoch_start"] += 1
            metadata["total_microbatches"] += 1

            inputs = _extract_into_tensor(train_loader, batch_indices, device=device)
            with torch.no_grad():
                targets = lm_head(_extract_into_tensor(orig_hiddens, batch_indices, device=device, dtype=torch_dtype))
                if hasattr(model_to_tune.config, "final_logit_softcapping"):  # Gemma
                    fls = model_to_tune.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls

            outputs = model_to_tune(inputs).logits
            loss = kl_div(outputs, targets.to(device=outputs.device, dtype=torch_dtype))

            metadata["loss_numerator"] += loss.item()
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")

            (loss / grad_accumulation_steps).backward()

            if layer._lora_A.grad is not None:
                metadata["23dj_gA"] = torch.linalg.norm(layer._lora_A.grad.data).item()
                metadata["23dj_gB"] = torch.linalg.norm(layer._lora_B.grad.data).item()
            if hasattr(layer, "input_low") and layer.input_low.grad is not None:
                metadata["23dj_gIL"] = torch.linalg.norm(layer.input_low.grad.data).item()
                metadata["23dj_gIR"] = torch.linalg.norm(layer.input_range.grad.data).item()
            if hasattr(layer, "scale") and layer.scale.grad is not None:
                metadata["23dj_gS"] = torch.linalg.norm(layer.scale.grad.data).item()

            if metadata["grad_steps_accumulated"] == grad_accumulation_steps:
                metadata["lr"] = get_lr(opt)
                opt.step()
                opt.zero_grad()
                # reset accumulated step and loss
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1
                metadata["aggregated_loss"] = metadata["loss_numerator"] / metadata["loss_denominator"]
                metadata["loss_numerator"] = metadata["loss_denominator"] = 0

                metadata["23dj_A"] = torch.linalg.norm(layer._lora_A.data).item()
                metadata["23dj_B"] = torch.linalg.norm(layer._lora_B.data).item()
                if hasattr(layer, "input_low"):
                    metadata["23dj_IL"] = torch.linalg.norm(layer.input_low.data).item()
                    metadata["23dj_IR"] = torch.linalg.norm(layer.input_range.data).item()
                else:
                    metadata["23dj_S"] = torch.linalg.norm(layer.scale.data).item()

            if (
                args.print_every_steps
                and metadata["total_optimizer_steps"] % args.print_every_steps == 0
                and metadata["grad_steps_accumulated"] == 0
            ):
                print(
                    f"epoch {metadata['current_epoch']}\t",
                    f"\t| total updates = {metadata['total_optimizer_steps']}",
                    f"\tloss = {metadata['aggregated_loss']:.9f}",
                    f"\tlr = {metadata['lr']:.9f}",
                )

            if args.mlflow:
                names_to_log = [
                    "aggregated_loss",
                    "lm_eval_word_ppl",
                    "lm_eval_word_ppl_no_init",
                    "best_eval_perplexity",
                    "current_epoch",
                    "23dj_A",
                    "23dj_gA",
                    "23dj_B",
                    "23dj_gB",
                    "23dj_S",
                    "23dj_gS",
                    "23dj_IL",
                    "23dj_gIL",
                    "23dj_IR",
                    "23dj_gIR",
                ]
                log_data = OrderedDict(filter(lambda pair: pair[0] in names_to_log, metadata.items()))
                mlflow.log_metrics(log_data, step=metadata["total_microbatches"])

        save_checkpoint(model_to_tune, last_dir, ckpt_name)
        word_ppl = eval_on_wikitext(args.base_model, last_dir, args.eval_model_seqlen, args.finetune_dtype)
        print(word_ppl)
        metadata["lm_eval_word_ppl_no_init"] = metadata["lm_eval_word_ppl"] = word_ppl
        if word_ppl < metadata["best_eval_perplexity"]:
            print(f"New best lm_eval word perplexity = {word_ppl:.4f}")
            metadata["best_eval_perplexity"] = word_ppl
            metadata["best_step"] = metadata["total_optimizer_steps"]
            shutil.copy(last_dir / ckpt_name, ckpt_dir / ckpt_name)
            shutil.copy(last_dir / "results.json", ckpt_dir / "results.json")

        metadata["microbatches_since_epoch_start"] = 0
        metadata["current_epoch"] += 1


def print_memory_stats():
    print(f"GPU max memory allocated: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB.")
    print(f"GPU max memory reserved: {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB.")


def get_argument_parser():
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--nncf_ckpt_dir",
        type=str,
        required=False,
        help="path to quantized model",
    )

    # Data params
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="number of samples",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--eval_model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen on validation. By default is equal to model_seqlen.",
    )
    parser.add_argument("--skip_first_eval", action="store_true", default=None)

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="finetuning learning rate",
    )
    parser.add_argument(
        "--fq_lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=32,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size",
    )
    parser.add_argument(
        "--finetune_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to finetune the model",
    )

    # Logging params
    parser.add_argument("--mlflow", action="store_true", help="Whether to use mlflow or store locally.")
    parser.add_argument(
        "--print_every_steps",
        type=int,
        default=None,
        help="print training metrics once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for logging.")

    # Misc params
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=[None, "auto"],
        help="accelerate device map",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        default=False,
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)

    assert torch.cuda.is_available()
    set_seed(args.seed)
    model_name = Path(args.base_model).name.replace(".", "_")
    ROOT_MODEL_DIR = (Path.home() / ("MODEL_DIR")).resolve()
    assert ROOT_MODEL_DIR.exists()
    MODEL_DIR = ROOT_MODEL_DIR / model_name
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    exp_name = (
        args.exp_name
        if args.exp_name
        else f"{model_name[:5]}_lr{args.lr:.0e}_fqlr{args.fq_lr:.0e}_wd{args.weight_decay:.0e}_tune_all"
    )
    ckpt_dir = Path(args.nncf_ckpt_dir) / exp_name
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    log_filename = ckpt_dir / "tune.log"
    print("Log file: ", log_filename.resolve())
    sys.stdout.flush()
    with log_filename.open("w") as f, redirect_stdout(f), redirect_stderr(f):
        pprint.pprint(vars(args))
        if args.mlflow:
            mlflow.set_experiment("Tune FQLoRA")

        init_ppl = None
        init_ppl = eval_on_wikitext(
            args.base_model, Path(args.nncf_ckpt_dir), args.eval_model_seqlen, args.finetune_dtype
        )
        print("word ppl for int4 init", init_ppl)

        # get data
        train_dataloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.model_seqlen,
            use_fast_tokenizer=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
            model_id=args.base_model,
        )

        # create original model
        orig_model = get_model(
            args.base_model, args.finetune_dtype, args.device_map, trust_remote_code=args.trust_remote_code
        )
        device = "cuda"
        if not args.device_map:
            orig_model = orig_model.to(device)
        lm_head = deepcopy(orig_model.lm_head)

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, use_fast=args.use_fast_tokenizer, trust_remote_code=True
        )

        # cache logits to not store original model, except the last layer.
        CACHE_DIR = MODEL_DIR / "hiddens_cache"
        CACHE_DIR.mkdir(exist_ok=True, parents=True)
        orig_hiddens = get_orig_hiddens(orig_model, train_dataloader, args.model_seqlen, args.dataset, CACHE_DIR)

        # Load model with FQ and LoRA adapters
        quant_model = load_nncf_quantized_model(args.nncf_ckpt_dir, orig_model, tokenizer)
        print("NNCF model device=", quant_model.device)
        if not args.device_map:
            quant_model = quant_model.to(device)

        with mlflow.start_run(run_name=exp_name) as run:
            try:
                print(f"Run ID: {run.info.run_id}")
                mlflow.log_params(vars(args))
                finetune(
                    quant_model,
                    train_loader=train_dataloader,
                    orig_hiddens=orig_hiddens,
                    args=args,
                    device=device,
                    ckpt_dir=ckpt_dir,
                    lm_head=lm_head,
                    init_ppl=init_ppl,
                )
            finally:
                print_memory_stats()
                if args.mlflow:
                    print("Adding log to artifacts: ", log_filename)
                    mlflow.log_artifact(log_filename)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
                    mlflow.log_params({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})


if __name__ == "__main__":
    main(sys.argv[1:])
