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
import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from optimum.modeling_base import OptimizedModel
from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.common.logging.track_progress import track
from nncf.data.dataset import Dataset
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.advanced_parameters import AdvancedAWQParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricLoraQuantizer

warnings.filterwarnings("ignore", category=TracerWarning)


def get_wikitext2(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    """
    Loads and processes the Wikitext-2 dataset for training.

    :param num_samples: Number of samples to generate.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer to encode the text.
    :param device: Device to move the tensors to (e.g., 'cpu' or 'cuda').
    :return: A list of tensors containing the tokenized text samples.
    """
    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    limit = num_samples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    trainloader = []
    for _ in range(num_samples):
        # Crop a sequence of tokens of length seqlen starting at a random position
        i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
    return trainloader


def measure_perplexity(
    optimum_model: OptimizedModel,
    max_length: Optional[int] = None,
    limit: Optional[Union[int, float]] = None,
) -> float:
    """
    Measure perplexity on the Wikitext dataset, via rolling loglikelihoods for a given model.

    :param optimum_model: A model to be evaluated.
    :param max_length: The maximum sequence length for evaluation.
    :param limit: Limit the number of examples per task (only use this for testing).
        If <1, limit is a percentage of the total number of examples.
    :return: The similarity score as a float.
    """
    task = "wikitext"
    print("#" * 50 + " Evaluate via lm-eval-harness " + "#" * 50)
    lm_obj = OptimumLM(pretrained=optimum_model, max_length=max_length)
    results = simple_evaluate(lm_obj, tasks=[task], limit=limit, log_samples=False)
    return results["results"][task]["word_perplexity,none"]


def evaluate_with_vllm(
    checkpoint_dir: Union[str, Path],
    tasks: list[str],
    tensor_parallel_size: int = 2,
    dtype: str = "auto",
    fewshot_as_multiturn: bool = True,
    apply_chat_template: bool = True,
    batch_size: str = "auto",
    limit: Optional[Union[int, float]] = None,
    cuda_devices: str = "1,2",
) -> dict:
    """
    Evaluate a model using lm_eval with vLLM backend in a subprocess.

    Runs evaluation in a subprocess to ensure CUDA_VISIBLE_DEVICES is set before
    CUDA initialization, which is required for proper GPU memory management.

    :param checkpoint_dir: Path to the model checkpoint directory.
    :param tasks: List of evaluation tasks (e.g., ["gsm8k"]).
    :param tensor_parallel_size: Number of GPUs for tensor parallelism.
    :param dtype: Data type for the model (e.g., "auto", "float16", "bfloat16").
    :param fewshot_as_multiturn: Whether to use fewshot examples as multi-turn conversation.
    :param apply_chat_template: Whether to apply the chat template.
    :param batch_size: Batch size for evaluation ("auto" for automatic).
    :param limit: Limit the number of examples per task (only use this for testing).
    :param cuda_devices: Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES.
    :return: Dictionary containing evaluation results.
    """
    print("#" * 50 + " Evaluate via lm-eval-harness (vLLM) " + "#" * 50)

    checkpoint_path = str(checkpoint_dir)
    model_args = f"pretrained={checkpoint_path},dtype={dtype},tensor_parallel_size={tensor_parallel_size}"

    cmd = [
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(checkpoint_dir / "lm_eval_results"),
    ]
    if fewshot_as_multiturn:
        cmd.append("--fewshot_as_multiturn")
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    # Set CUDA_VISIBLE_DEVICES in subprocess environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    print(f"Running: CUDA_VISIBLE_DEVICES={cuda_devices} {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        msg = f"lm_eval failed with return code {result.returncode}"
        raise RuntimeError(msg)

    # Parse results from the output JSON file
    results_dir = checkpoint_dir / "lm_eval_results"
    # Find the most recent results file
    result_files = list(results_dir.glob("**/results.json"))
    if not result_files:
        msg = f"No results.json found in {results_dir}"
        raise FileNotFoundError(msg)
    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
    with open(latest_result) as f:
        results = json.load(f)

    # Print results summary
    print("\nEvaluation Results:")
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if not metric.endswith("_stderr"):
                print(f"  {metric}: {value}")

    return results


@torch.no_grad()
def calc_hiddens(model: nn.Module, dataloader: list[Tensor]) -> list[Tensor]:
    """
    Calculate the hidden states for each input in the dataloader using the given model.

    :param model: The model used to calculate the hidden states.
    :param dataloader: The dataloader providing the inputs to the model.
    :return: A list of hidden states for each input in the dataloader.
    """
    orig_hiddens = []
    for data in track(dataloader, description="Calculating original hiddens"):
        model_input = get_model_input(data)
        orig_hiddens.append(model.model(**model_input).last_hidden_state)
    torch.cuda.empty_cache()
    return orig_hiddens


def get_model_input(input_ids: Tensor) -> dict[str, Tensor]:
    """
    Prepares the model input dictionary with input IDs, attention mask, and position IDs.

    :param input_ids: Tensor containing the input IDs.
    :return: A dictionary with keys "input_ids", "attention_mask", and "position_ids",
        each mapping to their respective tensors.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def kl_div(student_hiddens: torch.Tensor, teacher_hiddens: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence loss between the student and teacher hidden states.
    The input tensors are expected to have the same shape, and the last dimension represents the number of classes.

    :param student_hiddens: The hidden states from the student model.
    :param teacher_hiddens: The hidden states from the teacher model.
    :returns: The computed KL divergence loss.
    """
    num_classes = student_hiddens.shape[-1]
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, num_classes), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def set_trainable(model: nn.Module, lora_lr: float, fq_lr: float) -> list[dict[str, Any]]:
    """
    Sets the trainable parameters of the model for quantization-aware training with LoRA (Low-Rank Adaptation).

    This function disables gradients for all parameters in the model, then selectively enables gradients for
    specific quantizers (AsymmetricLoraQuantizer, SymmetricLoraQuantizer) that have 4-bit quantization.
    It collects the trainable parameters and adapters from these quantizers and returns them in a format
    suitable for an optimizer.

    :param model: The model to be trained.
    :param lora_lr: Learning rate for the LoRA adapters.
    :param fq_lr: Learning rate for the quantizer scales.
    :return: A list of dictionaries containing the parameters to be optimized and their corresponding learning rates.
    """
    model.requires_grad_(False)
    scales_to_train = []
    adapters_to_train = []
    hook_storage = get_hook_storage(model)
    for _, module in hook_storage.named_hooks():
        if isinstance(module, (AsymmetricLoraQuantizer, SymmetricLoraQuantizer)) and (module.num_bits in [4, 2]):
            module.enable_gradients()
            params = module.get_trainable_params()
            adapters = module.get_adapters()
            adapters_to_train.extend(adapters.values())
            scales_to_train.extend(param for name, param in params.items() if name not in adapters)

    params = list(model.parameters())
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    all_param = sum(p.numel() for p in params)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    model.train()
    return [{"params": adapters_to_train, "lr": lora_lr}, {"params": scales_to_train, "lr": fq_lr}]


def save_checkpoint(model: nn.Module, ckpt_file: Path, model_state: bool = True) -> None:
    """
    Stores the current state of a quantized model to a checkpoint file.

    :param model: The model whose state will be saved to checkpoint.
    :param ckpt_file: Path to store the checkpoint file.
    :param model_state: Whether to save the complete model weights in addition to NNCF state. Required when using
        AWQ method which fuses scaling factors into weights. When False, only NNCF configuration and state are saved,
        as they're maintained separately from the model's weights.
    """
    hook_storage = get_hook_storage(model)
    ckpt = {"nncf_state_dict": hook_storage.state_dict(), "nncf_config": nncf.torch.get_config(model)}
    if model_state:
        ckpt["model_state"] = model.state_dict()
    torch.save(ckpt, ckpt_file)


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    model = load_from_config(model, ckpt["nncf_config"])
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


@torch.no_grad()
def export_to_openvino(pretrained: str, ckpt_file: Path, ir_dir: Path) -> OVModelForCausalLM:
    """
    Create a wrapper of OpenVINO model from the checkpoint for evaluation on CPU via WWB.

    :param pretrained: The name or path of the pretrained model.
    :param ckpt_file: The path to the checkpoint file to load the model weights and NNCF configurations.
    :param last_dir: The directory where the OpenVINO model will be saved.
    :return: A wrapper of OpenVINO model ready for evaluation.
    """
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float32, device_map="cpu")
    model_to_eval = load_checkpoint(model_to_eval, ckpt_file)
    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.DQ)
    export_from_model(model_to_eval, ir_dir, device="cpu")
    return OVModelForCausalLM.from_pretrained(
        model_id=ir_dir,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=True,
    )


def limit_type(astr: str):
    value = float(astr)
    if value < 0 or value > 1:
        msg = "value not in range [0,1]"
        raise argparse.ArgumentTypeError(msg)
    return value


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained",
        type=str,
        default="Qwen/Qwen3-8B",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output",
        help="Path to the directory for storing logs, tuning checkpoint, compressed model, validation references.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to start from previously saved checkpoint. If not specified or checkpoint does not exist, "
        "start from scratch by post-training weight compression initialization.",
    )
    parser.add_argument("--lora_rank", type=int, default=64, help="Rank of lora adapters")
    parser.add_argument(
        "--basic_init",
        action="store_true",
        help="Whether to initialize quantization with basic min-max round-to-nearest schema. By default, advanced "
        "data-aware post-training methods are used: AWQ + Scale Estimation. These methods typically provide better "
        "accuracy, but require a calibration dataset and additional initialization time "
        "(~20 sec for 1B and ~80 sec for 8B models).",
    )

    # Data params
    parser.add_argument("--num_train_samples", type=int, default=512, help="Number of training samples")
    parser.add_argument("--train_seqlen", type=int, default=512, help="Train data context length.")
    parser.add_argument("--eval_seqlen", type=int, default=2048, help="Evaluation data context length.")
    parser.add_argument(
        "--limit",
        type=limit_type,
        default=None,
        help="A percentage of the total number of examples for evaluation. "
        "Should be on the range [0,1]. If None, all samples will be used.",
    )

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning. "
        "For larger models (over 3 billion parameters), a learning rate of 5e-5 is recommended.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=1,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    return parser


def main(argv) -> float:
    """
    Fine-tunes the specified model and returns the difference between initial and best validation perplexity in Torch,
    and the test perplexity for best model exported to OpenVINO.
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    assert torch.cuda.is_available()
    transformers.set_seed(42)
    device = "cuda"
    torch_dtype = torch.bfloat16
    compression_config = dict(
        mode=CompressWeightsMode.INT4_ASYM,
        group_size=64,
        awq=not args.basic_init,
        scale_estimation=not args.basic_init,
        compression_format=CompressionFormat.FQ_LORA,
    )
    pprint({"CLI arguments": vars(args), "Major compression parameters": compression_config})
    compression_config["advanced_parameters"] = AdvancedCompressionParameters(
        awq_params=AdvancedAWQParameters(prefer_data_aware_scaling=not args.basic_init),
        # scale_estimation_params=AdvancedScaleEstimationParameters(subset_size=-1, initial_steps=10, scale_steps=10),
        lora_adapter_rank=args.lora_rank,
    )
    # Configure output and log files.
    output_dir = Path(args.output_dir)
    tensorboard_dir = output_dir / "tb" / datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    last_dir = output_dir / "last"
    if not args.resume:
        shutil.rmtree(last_dir, ignore_errors=True)
    for path in [output_dir, tensorboard_dir, last_dir]:
        path.mkdir(exist_ok=True, parents=True)
    ckpt_file = last_dir / "nncf_checkpoint_svd_lora_se.pth"
    hidden_file = output_dir / "hiddens.pth"
    print(f"To visualize the loss and validation metrics, open Tensorboard using the logs from: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with absorbable LoRA")

    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Prepare training and calibration data
    train_loader = get_wikitext2(
        num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
    )
    if args.basic_init:
        example_input = {k: v.to(device) for k, v in model.dummy_inputs.items()}
        dataset = Dataset([example_input])
    else:
        calib_loader = get_wikitext2(num_samples=128, seqlen=128, tokenizer=tokenizer, device=device)
        dataset = Dataset(map(get_model_input, calib_loader))

    # Pre-compute hiddens of teacher model for distillation loss.
    if hidden_file.exists():
        orig_hiddens = torch.load(hidden_file, weights_only=False, map_location="cpu")
    else:
        orig_hiddens = calc_hiddens(model, train_loader)
        torch.save(orig_hiddens, hidden_file)

    # Create or load model to tune with Fake Quantizers and absorbable LoRA adapters.
    if args.resume and ckpt_file.exists():
        model = load_checkpoint(model, ckpt_file)
    else:
        model = compress_weights(model, dataset=dataset, **compression_config)
        save_checkpoint(model, last_dir / "nncf_checkpoint_svd_lora_se.pth", model_state=not args.basic_init)

    from nncf_layerwise_ptq_tuner import ScaleTuner

    model.requires_grad_(False)
    tuner = ScaleTuner(model)

    # Find optimal LR for LoRA params
    # result = tuner.lr_find(
    #     layer_pattern="gate_proj",  # First matching layer
    #     min_lr=1e-7,
    #     max_lr=1e6,  # Wide range for your case
    #     num_steps=100,
    #     loss_type="nmse",  # Better gradient signal
    #     param_type="lora",  # Test LoRA params specifically
    # )

    # Result contains:
    #   suggested_lr - where loss decreases fastest
    #   safe_lr - 1/10 of min loss point (conservative)
    #   lr_at_min_loss - LR at minimum loss

    # Plot results (requires matplotlib)
    # tuner.plot_lr_find(result, save_path="lr_find.png")

    # Then tune with the found LR:
    tuner.tune(
        tb,
        # learning_rate_lora=0,  # result["suggested_lr"], 2e-3
        learning_rate_lora=0,
        loss_type="nmse",
        num_steps=5000,
        scheduler_type_scale="constant",  # warmup + cosine annealing
        scheduler_type_lora="constant",  # no decay
        # layer_patterns=["layers:4:mlp:down_proj", "layers:15:mlp:down_proj"],
        learning_rate_2bit=1e-2,
        # learning_rate_2bit=0,
        # layer_patterns=["layers:20:mlp:down_proj"],
        learning_rate_4bit=5,
        # layer_patterns=["layers:0:mlp:gate_proj"],
        # [1/1] post_hooks.model:layers:0:mlp:gate_proj:weight__0.0
        # Type: sym_lora, Bits: 4, LR: 100000
        # INFO:nncf:Autograd-based quantization enabled
        early_stop_patience=1000,
        warmup_steps=0,
        min_lr_ratio=0.1,  # Anneal down to 1% of max LR
        restore_best=True,
        use_autograd_quantize=True,
        outlier_ratio=0,
    )
    tuner.print_summary()
    save_checkpoint(model, last_dir / "nncf_checkpoint_svd_lora_se_tune_scales.pth", model_state=not args.basic_init)

    fq_lr = args.lr / 10
    weight_decay = args.lr
    param_to_train = set_trainable(model, lora_lr=args.lr, fq_lr=fq_lr)
    opt = torch.optim.SGD(param_to_train, weight_decay=weight_decay)

    # Run tuning with distillation loss and validation after each epoch.
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_steps = 0
    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        for indices in track(batch_indices_epoch, description=f"Train epoch {epoch}"):
            indices = indices.tolist()

            def form_batch(inputs: list[Tensor], model_input: bool):
                batch = torch.cat([inputs[i] for i in indices], dim=0)
                return get_model_input(batch) if model_input else batch.to(device=device, dtype=torch_dtype)

            # Compute distillation loss between logits of the original model and the model with FQ + LoRA.
            inputs = form_batch(train_loader, model_input=True)
            with torch.no_grad():
                targets = model.lm_head(form_batch(orig_hiddens, model_input=False))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma has post-processing after lm_head
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls
            outputs = model(**inputs).logits
            loss = kl_div(outputs, targets.to(dtype=torch_dtype, device=device))

            # Perform an optimization step after accumulating gradients over multiple minibatches.
            loss_numerator += loss.item()
            grad_steps += 1
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)
            (loss / grad_accumulation_steps).backward()
            if grad_steps == grad_accumulation_steps:
                opt.step()
                opt.zero_grad()
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
                total_steps += 1
                tb.add_scalar("loss", aggregated_loss, total_steps)
        if epoch == 0:
            save_checkpoint(model, last_dir / "nncf_checkpoint_after_first_epoch.pth", model_state=not args.basic_init)
        else:
            save_checkpoint(model, ckpt_file, model_state=not args.basic_init)

    model = nncf.strip(model, strip_format=nncf.StripFormat.IN_PLACE)
    model.save_pretrained(last_dir / "stripped")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    tokenizer.save_pretrained(last_dir / "stripped")

    # Evaluate using lm_eval with vLLM backend (runs in subprocess for clean CUDA state)
    # del model
    # del opt
    # del orig_hiddens
    # del train_loader
    # del dataset
    # gc.collect()
    # torch.cuda.synchronize()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    # stripped_dir = last_dir / "stripped"
    # eval_results = evaluate_with_vllm(
    #     checkpoint_dir=stripped_dir,
    #     tasks=["gsm8k"],
    #     tensor_parallel_size=2,
    #     dtype="auto",
    #     fewshot_as_multiturn=True,
    #     cuda_devices="1,2",
    #     apply_chat_template=True,
    #     batch_size="auto",
    #     limit=args.limit,
    # )
    # gsm8k_acc = eval_results["results"]["gsm8k"]["exact_match,strict-match"]
    # tb.add_scalar("gsm8k_exact_match", gsm8k_acc, 0)
    # print(f"GSM8K exact match accuracy: {gsm8k_acc:.4f}")

    # del model
    # Export the best tuned model to OpenVINO and evaluate it using LM-Evaluation-Harness.
    # model_for_eval = export_to_openvino(args.pretrained, ckpt_file, ckpt_file.parent)
    # ov_perplexity = measure_perplexity(model_for_eval, args.eval_seqlen, args.limit)
    # tb.add_scalar("ov_perplexity", ov_perplexity, 0)
    # print(
    #     f"The finetuned model has been exported to OpenVINO and saved to: {last_dir}\n"
    #     f"The word perplexity on wikitext (test) = {ov_perplexity:.4f}"
    # )
    # return ov_perplexity
    # return gsm8k_acc


if __name__ == "__main__":
    main(sys.argv[1:])
