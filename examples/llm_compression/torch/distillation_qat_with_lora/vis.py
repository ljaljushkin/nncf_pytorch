from pathlib import Path
import re
import pandas as pd
import json

search_dir = Path("/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/distillation_qat_with_lora")

data = []

# Iterate over all .json files in the directory (recursively)
for json_file in search_dir.rglob('**/results_*.json'):
    epoch = json_file.name.split('_')[-1].split('.')[0]
    if all(c in "0123456789" for c in epoch):
        epoch = int(epoch)
    # print(json_file)
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

df = pd.DataFrame(data, columns=['Name', 'Epoch', 'wikitext', 'hellaswag', 'gsm8k_sm', 'gsm8k_fm'])
df = df[
    (df['Name'].str.contains('asym') | df['Name'].str.endswith('_repro_r256')) &
    ~df['Name'].str.contains('Qwen_Qwen2_5-1_5B-Instruct') &
    df['Epoch'].isin([0, 1, 2, 3, 9, 777])
    # &
    # orig_df['model'].str.contains('Phi') &
    # ~df['Name'].isin(['winogrande']) &
    # ~df['wikitext'].isnull()
    # orig_df['model'].isin(['Phi-3-mini-4k-instruct', 'microsoft_Phi-3-mini-4k-instruct'])
    # df = df[df['Name'].str.startswith('out_Qwen_Qwen2_5-1_5B-Instruct_repro')]
    # df = df[df['Epoch'].notna()]
]
df = df.pivot_table(index=['Name', 'Epoch'], values=['wikitext', 'hellaswag', 'gsm8k_fm', 'gsm8k_sm']).reset_index()
print(df.tail(2))

# 2. Isolate the baseline values (where Epoch == 0) for each 'Name'
# We only select the columns needed for the baseline calculation
baseline_df = df[df['Epoch'] == 777][['Name', 'wikitext', 'hellaswag', 'gsm8k_fm', 'gsm8k_sm']].copy()

# Rename columns to avoid conflicts when merging
baseline_df.rename(columns={
    'wikitext': 'wikitext_ref',
    'hellaswag': 'hellaswag_ref',
    'gsm8k_fm': 'gsm8k_fm_ref',
    'gsm8k_sm': 'gsm8k_sm_ref',
}, inplace=True)

# 3. Merge the baseline values back to the original DataFrame
# A 'left' merge ensures all original rows are kept.
# Rows with a 'Name' that has no Epoch 0 (like 'reference') will get NaN.
df = pd.merge(df, baseline_df, on='Name', how='left')

# 4. Apply the formula to create the 'avg relative error' column
# The formula is applied element-wise across the columns.

error1 = (df['wikitext'] - df['wikitext_ref']) / df['wikitext_ref']
error2 = (df['gsm8k_fm_ref'] - df['gsm8k_fm']) / df['gsm8k_fm_ref']
error3 = (df['gsm8k_sm_ref'] - df['gsm8k_sm']) / df['gsm8k_sm_ref']
error4 = (df['hellaswag_ref'] - df['hellaswag']) / df['hellaswag_ref']

# Average the two errors
df['wiki_rel_err'] = error1
df['gsm8k_fm_rel_err'] = error2
df['gsm8k_sm_rel_err'] = error3
df['hellaswag_rel_err'] = error4
# df['avg_rel_err'] = (error1 + error2 + error3 + error4) / 4
df['3avg_rel_err'] = (error1 + error2 + error4) / 3

baseline_df.rename(columns={
    'wikitext': 'wikitext_ref',
    'hellaswag': 'hellaswag_ref',
    'gsm8k_fm': 'gsm8k_fm_ref',
    'gsm8k_sm': 'gsm8k_sm_ref',
}, inplace=True)


# 5. Final Cleanup (Optional): Drop the temporary baseline columns
df.drop(columns=['wikitext_ref', 'gsm8k_fm_ref', 'hellaswag_ref','gsm8k_sm_ref'], inplace=True)
df.sort_values(by=["Epoch", "Name"], inplace=True)

df['Name'] = df['Name'].str.replace('_alpha_gs64_asym', '_DAWQ_SE_INIT')
df['Name'] = df['Name'].str.replace('out_', '')
df['Name'] = df['Name'].str.replace('_repro_r256', '_RTN_INIT')

# 2. Define the transformation rules
# rules = {
#     '_alpha_gs64_asym': 'DAWQ_SE',
#     '_repro_r256': 'RTN'
# }
# # 3. Initialize the new column with a default value
# # Using pd.NA is better than an empty string for "truly missing"
# df['Init mode'] = pd.NA
# for substring, init_mode_value in rules.items():
#     # Create a boolean mask for rows containing the substring
#     # na=False ensures that any NaN values in 'Name' are treated as False
#     mask = df['Name'].str.contains(substring, na=False)
#     # For the rows that match the mask:
#     # a) Set the value in the 'Init mode' column
#     df.loc[mask, 'Init mode'] = init_mode_value
#     # b) Remove the substring from the 'Name' column
#     # We use .loc[mask, 'Name'] to only apply the expensive replace operation
#     # to the subset of rows that need it.
#     df.loc[mask, 'Name'] = df.loc[mask, 'Name'].str.replace(substring, '')


df.to_csv('/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/distillation_qat_with_lora/tmp.csv')
print(df.tail(2))


def dump_pivot_to_excel(df, output_path: Path):
    # to have all columns, not only pivot's values, but also index one.
    print(df.columns)
    # df = df[~df['Epoch'].isin([777])]
    # df.drop(columns=['wikitext', 'hellaswag', 'gsm8k_fm', 'gsm8k_sm'], inplace=True)
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="all", index=False)
    (max_row, max_col) = df.shape
    workbook = writer.book
    worksheet = writer.sheets["all"]

    integer_format = workbook.add_format({'num_format': '0'})
    worksheet.set_column("B:B", 18, integer_format)
    float_format_3dp = workbook.add_format({"num_format": '0.000'})
    worksheet.set_column("C:X", 18, float_format_3dp)
    float_format_2dp = workbook.add_format({"num_format": '0.00'})
    worksheet.set_column("F:F", 18, float_format_2dp)
    float_format_1dp = workbook.add_format({"num_format": '0.0'})
    worksheet.set_column("G:X", 18, float_format_1dp)

    percent_format = workbook.add_format({'num_format': '0.0%'})
    worksheet.set_column('G:X', None, percent_format)

    col_names = [{"header": col_name} for col_name in df.columns]
    worksheet.add_table(
        0,
        0,
        df.shape[0],
        df.shape[1] - 1,
        {
            "columns": col_names,
            # 'style' = option Format as table value and is case sensitive
            # (look at the exact name into Excel)
            "style": None,
        },
    )

    # light_green='#63BE7B'
    # light_yellow='#FFEB84'
    # light_red='#F8696B'
    # higher_better_format = {"type": "3_color_scale", 'min_color': light_red, 'max_color': light_green, 'mid_color': light_yellow}
    # lower_better_format = {"type": "3_color_scale", 'min_color': light_green, 'max_color': light_red, 'mid_color': light_yellow}
    # max_row += 1
    # for letter in ['G', 'H', 'I', 'J', 'K']:
    #     worksheet.conditional_format(f"{letter}2:{letter}{max_row}", lower_better_format)
    # for letter in ['D','C', 'E']:
    #     worksheet.conditional_format(f"{letter}2:{letter}{max_row}", higher_better_format)
    # green_format = workbook.add_format({'bg_color':'#9BBB59'})
    green_format = workbook.add_format({'font_color':'#008000'})
    worksheet.conditional_format(f'G2:G{max_row}' , {'type': 'cell', 'criteria': '<', 'value': 0.05, 'format':  green_format})

    all_columns = df.columns.tolist()
    # Hide the column using its index
    for col_idx in range(2,6):
        # col_idx = all_columns.index(col_name)
        worksheet.set_column(col_idx, col_idx, None, None, {'hidden': True})
    col_idx = all_columns.index('gsm8k_sm_rel_err')
    worksheet.set_column(col_idx, col_idx, None, None, {'hidden': True})

    worksheet.autofit()
    workbook.close()
    print("Path to parsed results: ", output_path)

dump_pivot_to_excel(df, '/local_ssd2/nlyalyus/projects/nncf/examples/llm_compression/torch/distillation_qat_with_lora/tmp.xlsx')
df.tail(2)



