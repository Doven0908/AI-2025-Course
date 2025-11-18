#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/modules/selfrag-lite"

DEFAULT_MODEL="selfrag/selfrag_llama2_7b"
DEFAULT_TASK="asqa"
DEFAULT_INPUT="eval_data/asqa_eval_gtr_top100.json"
DEFAULT_OUTPUT="self_rag_output_demo"

echo "[selfrag-lite] Running long-form evaluation"
python run_long_form_static.py \
  --model_name "${MODEL_NAME:-$DEFAULT_MODEL}" \
  --task "${TASK:-$DEFAULT_TASK}" \
  --input_file "${INPUT_FILE:-$DEFAULT_INPUT}" \
  --output_file "${OUTPUT_DIR:-$DEFAULT_OUTPUT}" \
  --ndocs "${NDOCS:-5}" \
  --threshold "${THRESHOLD:-0.2}" \
  --max_depth "${MAX_DEPTH:-7}" \
  --mode "${MODE:-always_retrieve}" \
  "$@"
