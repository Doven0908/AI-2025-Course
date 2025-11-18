#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/modules/comprehensive"

echo "[comprehensive] Launching evaluation with config.yaml"
python main.py "$@"
