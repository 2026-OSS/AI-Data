#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_DIR="$ROOT_DIR/yolov5"
ROBOFLOW_VERSION="${ROBOFLOW_VERSION:-2}"
DATASET_NAME="AI-Picture-Book-Object-Detection-${ROBOFLOW_VERSION}"
DATASET_DIR="$YOLO_DIR/$DATASET_NAME"
DATA_YAML="$DATASET_DIR/data.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export ROBOFLOW_VERSION DATASET_NAME

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
  echo "ROBOFLOW_API_KEY is required."
  echo "Example: export ROBOFLOW_API_KEY='your_key_here'"
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -d "$YOLO_DIR" ]]; then
  git clone https://github.com/ultralytics/yolov5 "$YOLO_DIR"
fi

cd "$YOLO_DIR"
if [[ "${INSTALL_CUDA_TORCH:-1}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip install -q roboflow

if [[ ! -f "$DATA_YAML" ]]; then
  "$PYTHON_BIN" - <<'PY'
import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("2026-oss").project("ai-picture-book-object-detection")
project.version(int(os.environ.get("ROBOFLOW_VERSION", "2"))).download("yolov5")
PY
fi

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

dataset_name = os.environ["DATASET_NAME"]
yaml_path = Path(dataset_name) / "data.yaml"
data = yaml_path.read_text()
replacements = {
    "train: train/images": f"train: ./{dataset_name}/train/images",
    "val: valid/images": f"val: ./{dataset_name}/valid/images",
    "test: test/images": f"test: ./{dataset_name}/test/images",
    "train: ../train/images": f"train: ./{dataset_name}/train/images",
    "val: ../valid/images": f"val: ./{dataset_name}/valid/images",
    "test: ../test/images": f"test: ./{dataset_name}/test/images",
}
for old, new in replacements.items():
    data = data.replace(old, new)
yaml_path.write_text(data)
PY

if [[ ! -f yolov5s.pt ]]; then
  "$PYTHON_BIN" - <<'PY'
from urllib.request import urlretrieve
urlretrieve(
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "yolov5s.pt",
)
PY
fi

nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

"$PYTHON_BIN" train.py \
  --img "${IMG_SIZE:-640}" \
  --batch "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-100}" \
  --data "./${DATASET_NAME}/data.yaml" \
  --weights yolov5s.pt \
  --device "${CUDA_DEVICE:-0}" \
  --name "${RUN_NAME:-picture_book_yolov5_v${ROBOFLOW_VERSION}}"
