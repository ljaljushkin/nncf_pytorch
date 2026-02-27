export CUDA_VISIBLE_DEVICES=0
export GROUP_SIZE=64

# for BENCHMARK_MODE in CUDA TRITON COMPILE; do
for BENCHMARK_MODE in CUDA; do
    export BENCHMARK_MODE
    cd ~/projects/nncf2/examples/llm_compression/torch/distillation_qat_with_lora
    # export OPTIMIZED_BACKWARD=1
    # python3 main.py --num_train_samples 512 --epochs 1 | tee logs/${BENCHMARK_MODE}_opt.log 2>&1
    # export OPTIMIZED_BACKWARD=0
    python3 main.py --num_train_samples 512 --epochs 1 | tee logs/${BENCHMARK_MODE}.log 2>&1

    # cd ~/projects/nncf2
    # export OPTIMIZED_BACKWARD=1
    # PYTHONPATH=`pwd` python tools/benchmark_quantize_layers.py ${BENCHMARK_MODE}_opt.csv
    # export OPTIMIZED_BACKWARD=0
    # PYTHONPATH=`pwd` python tools/benchmark_quantize_layers.py ${BENCHMARK_MODE}.csv
done

# nsys profile --trace=cuda,nvtx --output=cuda_trace --force-overwrite true python3 main.py --num_train_samples 124 --epochs 1 | tee logs/cuda_asym_gs64_backup_none_no_reshape_all_bf16_NO_LORA_fwd_no_opt.log 2>&1
# nsys profile --trace=cuda,nvtx --output=triton_trace --force-overwrite true python3 main.py --num_train_samples 124 --epochs 1 | tee logs/triton_asym_gs64_backup_none_no_reshape_all_bf16_NO_LORA_fwd_no_opt.log 2>&1
# python3 main.py --num_train_samples 124 --epochs 1 | tee logs/$BENCHMARK_MODE.log 2>&1
# CUDA_VISIBLE_DEVICES=0 GROUP_SIZE=64 BENCHMARK_MODE=TRITON python main.py --num_train_samples 124 --epochs 1 | tee logs/triton_asym_gs64_backup_none_no_reshape_all_bf16_NO_LORA.log 2>&1
# CUDA_VISIBLE_DEVICES=0 GROUP_SIZE=64 BENCHMARK_MODE=COMPILE python main.py --num_train_samples 124 --epochs 1 | tee logs/compile_asym_gs64_backup_none_no_reshape_all_bf16_NO_LORA.log 2>&1


# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=CUDA python tools/benchmark_quantize_layers.py benchmark_cuda_no_lora_360m_bwd_opt_no_igrad.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=TRITON python tools/benchmark_quantize_layers.py benchmark_triton_no_lora_360m_no_igrad.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=REFERENCE python tools/benchmark_quantize_layers.py benchmark_ref_no_lora_360m.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=COMPILE python tools/benchmark_quantize_layers.py benchmark_compile_no_lora_360m.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=TRITON python tools/benchmark_quantize_layers.py benchmark_triton_tmp_clean_wall.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=COMPILE python tools/benchmark_quantize_layers.py benchmark_compile_tmp_clean_wall.csv
# PYTHONPATH=`pwd` GROUP_SIZE=64  BENCHMARK_MODE=REFERENCE python tools/benchmark_quantize_layers.py benchmark_ref_tmp_clean_wall.csv