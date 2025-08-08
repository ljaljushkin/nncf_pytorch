#!/usr/bin/env python3

matmul_shapes = {
    "HuggingFaceTB/SmolLM-1.7B-Instruct": {(2048, 2048), (2048, 8192), (8192, 2048), (49152, 2048)},
    "Qwen/Qwen2.5-1.5B-Instruct": {(256, 1536), (1536, 1536), (1536, 8960), (8960, 1536), (151936, 1536)},
    "Qwen/Qwen2.5-3B-Instruct": {(256, 2048), (2048, 2048), (2048, 11008), (11008, 2048), (151936, 2048)},
    "google/gemma-2-2b": {(1024, 2304), (2048, 2304), (2304, 2048), (2304, 9216), (9216, 2304), (256000, 2304)},
    "meta-llama/Meta-Llama-3-8B-Instruct": {(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096), (128256, 4096)},
    "microsoft/Phi-3-mini-4k-instruct": {(3072, 3072), (3072, 8192), (9216, 3072), (16384, 3072), (32064, 3072)},
    "bigcode/starcoder2-3b": {(256, 3072), (3072, 3072), (3072, 12288), (12288, 3072), (49152, 3072)},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        (256, 1536),
        (1536, 1536),
        (1536, 8960),
        (8960, 1536),
        (151936, 1536),
    },
    "meta-llama/Llama-2-7b-chat-hf": {(4096, 4096), (4096, 11008), (11008, 4096), (32000, 4096)},
    "meta-llama/Llama-3.2-1B-Instruct": {(512, 2048), (2048, 2048), (2048, 8192), (8192, 2048), (128256, 2048)},
    "meta-llama/Llama-3.2-3B-Instruct": {(1024, 3072), (3072, 3072), (3072, 8192), (8192, 3072), (128256, 3072)},
    "microsoft/Phi-3.5-mini-instruct": {(3072, 3072), (3072, 8192), (9216, 3072), (16384, 3072), (32064, 3072)},
    "mistralai/Mistral-7B-v0.1": {(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096), (32000, 4096)},
    "stabilityai/stablelm-3b-4e1t": {(2560, 2560), (2560, 6912), (6912, 2560), (50304, 2560)},
    "tinyllama/tinyllama-1.1b-step-50k-105b": {(256, 2048), (2048, 2048), (2048, 5632), (5632, 2048), (32000, 2048)},
}

# Collect all unique shapes
all_shapes = set()
for model_shapes in matmul_shapes.values():
    all_shapes.update(model_shapes)

# Analyze shapes
THRESHOLD = 8_388_608  # 8M elements (8192 blocks × 1024 threads)

print(f"Analysis of matrix shapes vs {THRESHOLD:,} element threshold:\n")

large_shapes = []
safe_shapes = []

for shape in sorted(all_shapes):
    elements = shape[0] * shape[1]
    status = "LARGE" if elements > THRESHOLD else "safe"

    if elements > THRESHOLD:
        large_shapes.append((shape, elements))
    else:
        safe_shapes.append((shape, elements))

    print(f"{shape}: {elements:,} elements - {status}")

print(f"\n{'=' * 60}")
print("SUMMARY:")
print(f"{'=' * 60}")
print(f"Total unique shapes: {len(all_shapes)}")
print(f"Safe shapes (≤ {THRESHOLD:,}): {len(safe_shapes)}")
print(f"Large shapes (> {THRESHOLD:,}): {len(large_shapes)}")
print(f"Percentage of problematic shapes: {len(large_shapes) / len(all_shapes) * 100:.1f}%")

if large_shapes:
    print("\nPROBLEMATIC SHAPES:")
    print(f"{'Shape':<20} {'Elements':<12} {'Coverage':<10} {'Missing Elements'}")
    print(f"{'-' * 60}")
    for shape, elements in large_shapes:
        coverage_pct = (THRESHOLD / elements) * 100
        missing = elements - THRESHOLD
        print(f"{str(shape):<20} {elements:,<12} {coverage_pct:.1f}%      {missing:,}")

# Count occurrences across all models
print("\nMODEL ANALYSIS:")
print(f"{'=' * 60}")
models_with_large = 0
total_large_occurrences = 0

for model, shapes in matmul_shapes.items():
    large_in_model = [s for s in shapes if s[0] * s[1] > THRESHOLD]
    if large_in_model:
        models_with_large += 1
        total_large_occurrences += len(large_in_model)
        print(f"{model}: {len(large_in_model)} large shapes")
        for shape in large_in_model:
            elements = shape[0] * shape[1]
            print(f"  {shape}: {elements:,} elements")

print(f"\nModels with large shapes: {models_with_large}/{len(matmul_shapes)}")
print(f"Total large shape occurrences across all models: {total_large_occurrences}")
