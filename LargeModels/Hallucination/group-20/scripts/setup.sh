#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] Using repo root: $REPO_ROOT"
echo "[setup] Python version:"
python --version

if [[ ! -d ".venv" ]]; then
  echo "[setup] Creating virtual environment at .venv"
  python -m venv .venv
fi

echo "[setup] Activating virtual environment"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  # shellcheck disable=SC1091
  source ".venv/Scripts/activate"
else
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "[setup] Installing SELF-RAG official requirements"
pip install -r modules/selfrag-official/requirements.txt

echo "[setup] Installing shared research tooling"
pip install -U ragas lettucedetect vllm gradio

echo "[setup] Installing SelfCheckGPT module as editable package"
pip install -e modules/selfcheckgpt

echo "[setup] Done! You can now run scripts in the scripts/ directory."
