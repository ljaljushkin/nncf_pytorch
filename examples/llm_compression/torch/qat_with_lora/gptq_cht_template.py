import logging

import torch
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq import BaseQuantizeConfig
from datasets import load_dataset
from torch import Tensor
from transformers import AutoTokenizer
from auto_gptq import BaseQuantizeConfig
from transformers import AutoTokenizer


def get_alpaca_with_chat(maxlen: int, tokenizer, device: torch.device) -> list[Tensor]:
    # dataset = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    dataset = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=1000)  # Shuffle a buffer

    attempts = 0
    trainloader = []
    total_num_tokens = 0
    for example in dataset:
        attempts += 1
        if total_num_tokens >= 128 * 1024:  # NUM_SAMPLES:  # If streaming, limit checks
            print(
                f"Collected {len(trainloader)} samples by checking {attempts} samples, Avg length: {total_num_tokens // len(trainloader)}"
            )
            break

        instruction = example.get("instruction", "")
        context = example.get("input", "")
        response = example.get("output", "")
        if not instruction or not response:
            print("Skipping empty instruction and response")
            continue

        prompt = instruction
        if context:
            prompt += f"\n\n{context.strip()}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer([text])
        input_ids = torch.tensor(model_inputs.input_ids[:maxlen], dtype=torch.int).to(device)
        total_num_tokens += input_ids.numel()
        trainloader.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device)))

    return trainloader

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
# Specify paths and hyperparameters for quantization
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
    model_name_or_path=None,
    model_file_base_name="model",
)

model = AutoGPTQForCausalLM.from_pretrained(
    model_id, quantize_config, device_map="cuda", max_memory={i: "80GB" for i in range(1)}
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
data = get_alpaca_with_chat(maxlen=8192, tokenizer=tokenizer, device="cuda")
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
model.quantize(data, cache_examples_on_gpu=False)
model.save_pretrained('out_Qwen_Qwen2_5-1_5B-Instruct_GPTQ_chat_alpaca_128x1024')