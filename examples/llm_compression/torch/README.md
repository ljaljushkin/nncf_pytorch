# Quantization-aware tuning with mergable LoRA Adapters for improving accuracy of 4bit LLMs

This example demonstrates how to improve accuracy of Large Language Models (LLMs) with 4bit weights by quantization-aware-training with mergable LoRA adapters.

The example includes the following steps:

- [add_FQ_with_LoRA.py](add_FQ_with_LoRA.py): This script creates an NNCF model with extended FakeQuantize (FQ) operations on the weights of all linear layers, except for the embedding and lm_head layers. This FQ includes mergeable LoRA Adapters and performs fake quantization in the following way: `dequantize(quantize(W + B @ A))`, where W is the original weight of the linear layer, and A and B are the LoRA adapters.
The compression part of the NNCF model is then saved in the NNCF checkpoint for tuning and evaluation. It is expected that the initial accuracy of such a model will be low, as it currently uses a data-free Round-To-Nearest quantization scheme. In the next step, accuracy will be significantly improved by tuning both the quantization scales and the LoRA adapters.

- [tune_fq_lora.py](tune_fq_lora.py): This script implements a tuning pipeline with distillation loss. The teacher model is the original bfloat16 model, while the student model includes FQ operations. The training dataset is based on the training portion of the `wikitext-2-raw-v1` dataset, consisting of 1024 samples of length 1024. Validation is performed at the end of each epoch on the test split of `wikitext-2-raw-v1`.
Tuning for 32 epochs on a single A100 card takes around 4 hours for 1.7B models, approximately 6 hours for 3B models, and about 12 hours for 8B models. The most significant accuracy improvement is typically achieved within the first 1-2 epochs.
- [wwb_eval.py](wwb_eval.py) or [patched lm_eval](https://github.com/ljaljushkin/lm-evaluation-harness/blob/nl/load_nncf_state/lm_eval/models/huggingface.py#L324): Evaluates NNCF checkpoints with tuned adapters and quantization scales using [WhoWhatBench](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python/who_what_benchmark) or [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The checkpoint is loaded on top of the original bfloat16 model. All FQ operations are automatically set up on the corresponding linear layers. To speed up evaluation on the GPU, the original bfloat16 weights are replaced by fake-quantized ones: `W = dequantize(quantize(W + B @ A))`. Evaluation is automated by the [eval.sh](eval.sh) and [eval_slm.sh](eval_slm.sh) scripts.

## Install requirements

To use this example:

- Create a separate Python* environment and activate it:

```bash
python3 -m venv env && source env/bin/activate
```

- Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
pip install ../../../
```

## Run Example

The example is fully automated. Just run the following script in the prepared Python environment to tune and evaluate `microsoft/Phi-3.5-mini-instruct`:

```bash
./tune.sh
```

The tuning convergence with the same hyperparameters was checked on 7 other models. To run the example for one of these models, simply uncomment the corresponding `BASE_MODEL` and `MODEL_NAME` in `tune.sh` and launch it.

Smaller models, like HuggingFaceTB/SmolLM-1.7B-Instruct, require slightly different hyperparameters (learning rate for quantization scales and LoRA adapters, maximum sequence length for evaluation). Just run the following script in the prepared Python environment for this and similar models:

```bash
./tune_smolm.sh
```
