# go over all files in s, find *.json file and create a DataFrame with the following structure:
    # Name | wikitext | hellaswag | gsm8k
    # out_meta-llama_Llama-3_2-3B-Instruct_repro_r256_alpha | 1 | 2 | 3 |
    # out_meta-llama_Llama-3_2-3B-Instruct_repro_r256_t2 | 3 | 2 | 3 |

# /local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/distillation_qat_with_lora/
    # out_meta-llama_Llama-3_2-3B-Instruct_repro_r256_alpha/
        # last/
            # results_hellaswag.json

# {
#     "results": {
#         "wikitext": {
#             "alias": "wikitext",
#             "word_perplexity,none": 13.160949160720696,

# {
#     "results": {
#         "hellaswag": {
#             "alias": "hellaswag",
#             "acc,none": 0.5163314080860386,

from pathlib import Path
import pandas as pd
import json

# Directory to search
search_dir = Path("/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/distillation_qat_with_lora")

# Prepare data list
data = []

# Iterate over all .json files in the directory (recursively)
for json_file in search_dir.rglob('**/results_*.json'):
    print(json_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    # Extract values, default to None if not present
    name = json_file.parent.parent.stem
    wikitext = content.get('results', {}).get('wikitext', {}).get('word_perplexity,none')
    hellaswag = content.get('results', {}).get('hellaswag', {}).get('acc,none,none')
    gsm8k_sm = content.get('results', {}).get('gsm8k', {}).get('exact_match,strict-match')
    gsm8k_fm = content.get('results', {}).get('gsm8k', {}).get('exact_match,flexible-extract')

    data.append({
        'Name': name,
        'wikitext': wikitext,
        'hellaswag': hellaswag,
        'gsm8k_sm': gsm8k_sm,
        'gsm8k_fm': gsm8k_fm
    })

# Create DataFrame
df = pd.DataFrame(data, columns=['Name', 'wikitext', 'hellaswag', 'gsm8k_sm', 'gsm8k_fm'])
df.pivot(index='Name', columns=['wikitext', 'hellaswag', 'gsm8k_sm', 'gsm8k_fm'])
print(df)