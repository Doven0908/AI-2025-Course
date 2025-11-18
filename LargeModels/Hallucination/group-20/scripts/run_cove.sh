#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/modules/cove"

OUTPUT_FILE="${OUTPUT_FILE:-hallucination_results_plus.json}"

echo "[cove] Running CoVe hallucination detection pipeline"
python main.py
python plus.py || true

if [[ -f "$OUTPUT_FILE" ]]; then
  echo "[cove] Results written to: $OUTPUT_FILE"
else
  echo "[cove] Finished, check modules/cove/ for outputs."
fi
