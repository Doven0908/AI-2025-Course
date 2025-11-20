set -ex

PROMPT_TYPE="cot"
MODEL_NAME_OR_PATH="/home/lijiakun25/models/Qwen2.5-3b-Instruct"
DATA_DIR="/home/lijiakun25/models/datasets/math500/test.jsonl"

export CUDA_VISIBLE_DEVICES="0"

# English open datasets
# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
DATA_NAME="math"
TOKENIZERS_PARALLELISM=false \
python3 -u MCTS_evaluate.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --data_dir ${DATA_DIR} \
    --dtype float16 \
    --num_of_models 4 \
    --max_func_call 80 \
