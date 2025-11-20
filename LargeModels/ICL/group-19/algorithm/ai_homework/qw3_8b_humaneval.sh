export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH=/home/lijiakun25/ai_homework:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1

python -m lm_eval \
  --include_path /home/lijiakun25/ai_homework/my_tasks \
  --model hf \
  --model_args "pretrained=/home/lijiakun25/models/LLaMA3-8b-Instr,dtype=float16,device_map=auto,trust_remote_code=true" \
  --tasks humaneval_local \
  --batch_size 16 \
  --log_samples \
  --confirm_run_unsafe_code \
  --output_path ./results_humaneval_local_full
