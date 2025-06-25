from pathlib import Path
import re
import pandas as pd
import json

# Directory to search
search_dir = Path("/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/distillation_qat_with_lora")

# Prepare data list
data = []

# Iterate over all .json files in the directory (recursively)
for json_file in search_dir.rglob('**/results_*.json'):
    # print(json_file)
    # match = re.search(r'\d+', json_file.name)
    epoch = json_file.name.split('_')[-1].split('.')[0]
    # epoch = None
    # if match:
    #     epoch = match.group()

    with open(json_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    # Extract values, default to None if not present
    name = json_file.parent.parent.stem
    wikitext = content.get('results', {}).get('wikitext', {}).get('word_perplexity,none')
    hellaswag = content.get('results', {}).get('hellaswag', {}).get('acc,none')
    gsm8k_sm = content.get('results', {}).get('gsm8k', {}).get('exact_match,strict-match')
    gsm8k_fm = content.get('results', {}).get('gsm8k', {}).get('exact_match,flexible-extract')

    data.append({
        'Name': name,
        'Epoch': epoch,
        'wikitext': wikitext,
        'hellaswag': hellaswag,
        'gsm8k_sm': gsm8k_sm,
        'gsm8k_fm': gsm8k_fm
    })

# Create DataFrame
df = pd.DataFrame(data, columns=['Name', 'Epoch', 'wikitext', 'hellaswag', 'gsm8k_sm', 'gsm8k_fm'])
# df = df[df['Name'].str.startswith('out_Qwen_Qwen2_5-1_5B-Instruct_repro')]
df = df[df['Epoch'].notna()]
print(df)

pt = df.pivot_table(index=['Name', 'Epoch'], values=['wikitext', 'hellaswag', 'gsm8k_sm', 'gsm8k_fm'])
print(pt)

# pt = df.groupby(['Name', 'Epoch']).agg({
#     'wikitext': 'min',
#     'gsm8k_fm': 'max',
#     'gsm8k_sm': 'max',
#     'hellaswag': 'max'
# }).reset_index()
# print(pt)

# modela_rows = pt[pt['Name'].str.startswith('out_Qwen_Qwen2_5-1_5B-Instruct_repro')]
# modela_rows = modela_rows.sort_values(by='Name')
# print(modela_rows)
pt.to_csv('/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/distillation_qat_with_lora/tmp.csv')
