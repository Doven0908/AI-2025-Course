set -ex

PROMPT_TYPE="qwen3-math-cot"
MODEL_NAME_OR_PATH="/home/lijiakun25/models/Qwen3-8b"
DATA_DIR="/home/lijiakun25/models/datasets/AIME25/aime2025.jsonl"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_FLASH_ATTN_DISABLE=1
export VLLM_USE_TRITON_FLASH_ATTN=0

# English open datasets
# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
DATA_NAME="aime25"
TOKENIZERS_PARALLELISM=false \
python3 -u eval.py \
    --eval_type cot \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --data_dir ${DATA_DIR} \
    --prompt_type ${PROMPT_TYPE} \
    --num_shots 0\
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 1 \
    --dtype float32 \
    --save_output_number 30 \
    --max_tokens_per_call 16384 \
    --model_enable_thinking false \
