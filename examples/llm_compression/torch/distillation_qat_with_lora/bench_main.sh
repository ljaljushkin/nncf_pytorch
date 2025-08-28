CUDA_VISIBLE_DEVICES=0 GROUP_SIZE=64 BENCHMARK_MODE=CUDA python main.py --num_train_samples 124 --epochs 1 | tee logs/cuda_asym_gs64_backup_none_no_reshape_all_bf16_NO_LORA_bwd_not_opt_fwd_no_opt.log 2>&1
CUDA_VISIBLE_DEVICES=0 GROUP_SIZE=64 BENCHMARK_MODE=TRITON python main.py --num_train_samples 124 --epochs 1 | tee logs/triton_asym_gs64_backup_none_no_reshape_all_bf16_NO_LORA.log 2>&1
CUDA_VISIBLE_DEVICES=0 GROUP_SIZE=64 BENCHMARK_MODE=COMPILE python main.py --num_train_samples 124 --epochs 1 | tee logs/compile_asym_gs64_backup_none_no_reshape_all_bf16_NO_LORA.log 2>&1

# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=CUDA python tools/benchmark_quantize_layers.py benchmark_cuda_no_lora_360m.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=TRITON python tools/benchmark_quantize_layers.py benchmark_triton_no_lora_360m.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=REFERENCE python tools/benchmark_quantize_layers.py benchmark_ref_no_lora_360m.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=COMPILE python tools/benchmark_quantize_layers.py benchmark_compile_no_lora_360m.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=TRITON python tools/benchmark_quantize_layers.py benchmark_triton_tmp_clean_wall.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=COMPILE python tools/benchmark_quantize_layers.py benchmark_compile_tmp_clean_wall.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=REFERENCE python tools/benchmark_quantize_layers.py benchmark_ref_tmp_clean_wall.csv