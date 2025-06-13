from pathlib import Path

import json
from optimum.intel.openvino import OVModelForCausalLM
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM

for name, half in [
    # ("out_HuggingFaceTB_SmolLM-1_7B-Instruct_cast", True),
    # ("out_meta-llama_Llama-3_2-1B-Instruct_cast", False),
    # ("out_meta-llama_Llama-3_2-3B-Instruct_cast", False),
    # ("out_google_gemma-2-2b-it_cast", False),
    # ("out_meta-llama_Meta-Llama-3-8B-Instruct_cast", False),
    # ("out_microsoft_Phi-3_5-mini-instruct_cast", False),
    # ("out_microsoft_Phi-3-mini-4k-instruct_cast", False),
    # ("out_mistralai_Mistral-7B-v0_3_cast", False),
    ("out_Qwen_Qwen2_5-3B-Instruct_cast", False)
]:
    max_length = 2048 if half else 4096
    ir_dir = Path('/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/qat_with_lora') / name / 'best'
    ir_xml = (ir_dir / 'openvino_model.xml')
    print(ir_xml)
    assert ir_xml.exists()
    ov_model = OVModelForCausalLM.from_pretrained(
        model_id=ir_dir,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=True,
    )
    task = "wikitext"
    lm_obj = OptimumLM(pretrained=ov_model, max_length=max_length)
    results = simple_evaluate(lm_obj, tasks=[task],  log_samples=False)
    ppl = results["results"][task]["word_perplexity,none"]
    print('word ppl: ', ppl)
    res_file = ir_dir / 'default_ov.json'
    with res_file.open('w') as json_file:
        json.dump({'word_ppl': ppl}, json_file, indent=4)
    print("results in file:", res_file)
    # with res_file.open('r') as json_file:
    #     print(json.load(json_file))