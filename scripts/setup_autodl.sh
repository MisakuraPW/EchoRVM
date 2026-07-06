#!/usr/bin/env bash
set -euo pipefail

echo "===== AutoDL Project Setup ====="

PROJECT_ROOT="$(pwd)"
DATASET_DIR="/root/autodl-tmp/datasets"
OUTPUT_DIR="/root/autodl-tmp/outputs"
CACHE_DIR="/root/autodl-tmp/cache"
LOG_DIR="/root/autodl-tmp/logs"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints"

mkdir -p "${DATASET_DIR}" "${OUTPUT_DIR}" "${CACHE_DIR}" "${LOG_DIR}" "${CHECKPOINT_DIR}"
mkdir -p "${DATASET_DIR}/EchoNet-Dynamic" "${DATASET_DIR}/CAMUS" "${DATASET_DIR}/EchoRisk"

echo "Project root: ${PROJECT_ROOT}"
echo "Dataset dir: ${DATASET_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Cache dir: ${CACHE_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"

echo
echo "Checking Python..."
python --version

echo
echo "Checking CUDA and PyTorch..."
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch check failed:", exc)
PY

if [[ "${1:-}" == "--install" ]]; then
  echo
  echo "Installing Python requirements from requirements.txt..."
  python -m pip install -r requirements.txt
fi

echo
echo "Run the detailed checker with:"
echo "  python tools/check_env.py"
echo "  python tools/check_dataset.py --dataset all"
echo "===== Setup Finished ====="
