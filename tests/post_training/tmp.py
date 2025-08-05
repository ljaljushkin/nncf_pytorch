import torch
from transformers import AutoTokenizer
from transformers import pipeline

model = "TinyLlama/TinyLlama_v1.1"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    "Who is the most famous athlete?",
    do_sample=False,
    # top_k=10,
    # num_return_sequences=1,
    repetition_penalty=1.5,
    # eos_token_id=tokenizer.eos_token_id,
    # max_length=500,
    max_new_tokens=128,
)
for seq in sequences:
    print(f"\n\n\nResult: {seq['generated_text']}")
