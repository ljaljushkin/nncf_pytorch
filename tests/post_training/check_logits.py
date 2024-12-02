
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import numpy as np

model_id = '/home/devuser/nlyalyus/projects/nncf/tests/post_training/tmp'
tokenizer = AutoTokenizer.from_pretrained('tinyllama/tinyllama-1.1b-step-50k-105b', trust_remote_code=True)
model = OVModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, load_in_8bit=False, compile=False, stateful=True
)
inputs = tokenizer("overfit", return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    min_new_tokens=2,
    max_new_tokens=2,
    output_scores=True,
    return_dict_in_generate=True,
    do_sample=False
)

# (some context before the actual answer) As we write in the docs, we only have scores for newly generated tokens.
# The concept of scores is different from logits: they are logits manipulated for token selection purposes.
# We don't manipulate the logits regarding the prompt in any way,
# and thus we technically don't have the scores of the prompt.

# Logits are an afterthought added recently, and we've decided to keep a 1:1 correspondence with the scores, for simplicity.
# There is also a way to obtain them without spending extra compute:
# run a forward pass with the prompt up to the penultimate token (i.e. input_ids[:, :-1]), keep logits and past_key_values
# pass past_key_values to generate, which skips the prefill stage

# log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

# Example 1: Print the scores for each token generated with Greedy Search
# outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)

# TODO: how to get logits for not selected (sequence) tokens?
# read https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1143
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)
print(transition_scores)
# input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# encoder-decoder models, like BART or T5.
input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
for i, (tok, score) in enumerate(zip(generated_tokens[0], transition_scores[0])):
    # | token | token string | log probability | probability
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
    # np.set_printoptions(precision=1)
    # topk_scores = np.sort(outputs.scores[i].squeeze_())[:-10]
    # print("topk_scores: ", topk_scores)
    # print("topk_probs: ", np.exp(topk_scores) * 100)
print(outputs)
print(tokenizer.batch_decode(generated_tokens))


# # Example 2: Reconstruct the sequence scores from Beam Search
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=5,
#     num_beams=4,
#     num_return_sequences=4,
#     return_dict_in_generate=True,
#     output_scores=True,
# )
# transition_scores = model.compute_transition_scores(
#     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
# )
# If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
# Tip 1: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
# use case, you might want to recompute it with `normalize_logits=True`.
# Tip 2: the output length does NOT include the input length
# output_length = np.sum(transition_scores.numpy() < 0, axis=1)
# length_penalty = model.generation_config.length_penalty
# reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
# print(np.allclose(outputs.sequences_scores, reconstructed_scores))
# print(reconstructed_scores)