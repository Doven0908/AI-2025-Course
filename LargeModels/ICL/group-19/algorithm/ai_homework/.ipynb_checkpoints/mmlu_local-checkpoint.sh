#!/bin/bash
# 本地 MMLU 评测脚本（双卡、HF 后端）

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export PYTHONPATH=/home/lijiakun25/ai_homework:$PYTHONPATH

python -m lm_eval \
  --include_path /home/lijiakun25/ai_homework/my_tasks \
  --model hf \
  --model_args '{
    "pretrained": "/home/lijiakun25/models/LLaMA3-8b-Instr",
    "dtype": "float16",
    "trust_remote_code": true,
    "device_map": "auto",
    "max_memory": {"0": "28GiB", "1": "28GiB"},
    "low_cpu_mem_usage": true
  }' \
  --tasks mmlu_local \
  --num_fewshot 2 \
  --batch_size 16 \
  --device cuda \
  --gen_kwargs '{"temperature":0.6,"do_sample":false}' \
  --log_samples \
  --output_path ./results_mmlu_local
