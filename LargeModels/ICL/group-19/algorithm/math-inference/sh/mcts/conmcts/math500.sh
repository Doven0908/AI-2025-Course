set -ex

PROMPT_TYPE="qwen3-math-cot"
MODEL_NAME_OR_PATH="/home/lijiakun25/models/Qwen3-8b"
DATA_DIR="/home/lijiakun25/models/datasets/math500/test_debug.jsonl"

# English open datasets
# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
DATA_NAME="math"
TOKENIZERS_PARALLELISM=false \
python3 -u eval.py \
    --eval_type conmcts \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --data_dir ${DATA_DIR} \
    --prompt_type ${PROMPT_TYPE} \
    --num_shots 0 \
    --n_sampling 2 \
    --dtype float32 \
    --num_of_models 2 \
    --max_func_call 2 \
    --model_enable_thinking true \
