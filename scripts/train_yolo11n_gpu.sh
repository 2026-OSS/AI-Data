#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ -z "${ROBOFLOW_API_KEY:-}" ]]; then
  echo "ROBOFLOW_API_KEY is required."
  echo "Example: export ROBOFLOW_API_KEY='your_key_here'"
  exit 1
fi

export ROBOFLOW_WORKSPACE="${ROBOFLOW_WORKSPACE:-2026-oss}"
export ROBOFLOW_PROJECT="${ROBOFLOW_PROJECT:-ai-picture-book-object-detection}"
export ROBOFLOW_VERSION="${ROBOFLOW_VERSION:-13}"
export MODEL_NAME="${MODEL_NAME:-yolo11n.pt}"
export RUN_NAME="${RUN_NAME:-picture_book_yolo11n_v${ROBOFLOW_VERSION}}"
export IMG_SIZE="${IMG_SIZE:-640}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-150}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"
export SEED="${SEED:-42}"
export PATIENCE="${PATIENCE:-30}"
export DATASET_NAME="AI-Picture-Book-Object-Detection-${ROBOFLOW_VERSION}"

cd "$ROOT_DIR"

if [[ "${INSTALL_CUDA_TORCH:-1}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
"$PYTHON_BIN" -m pip install -U ultralytics roboflow pyyaml pandas

nvidia-smi

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

root = Path.cwd()
dataset_name = os.environ["DATASET_NAME"]
dataset_dir = root / dataset_name
yaml_path = dataset_dir / "data.yaml"

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
print("gpu:", torch.cuda.get_device_name(0))

if not yaml_path.exists():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(os.environ["ROBOFLOW_PROJECT"])
    dataset = project.version(int(os.environ["ROBOFLOW_VERSION"])).download("yolov8")
    dataset_dir = Path(dataset.location)
    yaml_path = dataset_dir / "data.yaml"

with yaml_path.open("r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

data_yaml["train"] = str(dataset_dir / "train" / "images")
data_yaml["val"] = str(dataset_dir / "valid" / "images")
data_yaml["test"] = str(dataset_dir / "test" / "images")

with yaml_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

expected_classes = [
    "book_flower",
    "book_flowerpot",
    "book_monkey",
    "book_stone",
    "braille",
    "tactile_flower",
    "tactile_flowerpot",
    "tactile_monkey",
    "tactile_stone",
    "text",
]

names = data_yaml.get("names")
if isinstance(names, dict):
    actual_classes = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
else:
    actual_classes = names

if len(actual_classes) != len(expected_classes) or set(actual_classes) != set(expected_classes):
    raise RuntimeError(f"Unexpected classes: {actual_classes}")

model = YOLO(os.environ["MODEL_NAME"])
train_results = model.train(
    data=str(yaml_path),
    imgsz=int(os.environ["IMG_SIZE"]),
    epochs=int(os.environ["EPOCHS"]),
    batch=int(os.environ["BATCH_SIZE"]),
    name=os.environ["RUN_NAME"],
    project="runs/detect",
    device=int(os.environ["CUDA_DEVICE"]),
    seed=int(os.environ["SEED"]),
    deterministic=True,
    patience=int(os.environ["PATIENCE"]),
    cos_lr=True,
    cache=True,
)

save_dir = Path(train_results.save_dir)
best_model_path = save_dir / "weights" / "best.pt"
best_model = YOLO(str(best_model_path))
metrics = best_model.val(data=str(yaml_path), split="test", imgsz=int(os.environ["IMG_SIZE"]))

summary = {
    "model": os.environ["MODEL_NAME"],
    "dataset_version": int(os.environ["ROBOFLOW_VERSION"]),
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "mAP@0.5": float(metrics.box.map50),
    "mAP@0.5:0.95": float(metrics.box.map),
    "best_model_path": str(best_model_path),
    "data_yaml_path": str(yaml_path),
}

results_dir = root / "results"
results_dir.mkdir(exist_ok=True)
summary_path = results_dir / f"yolo11n_v{os.environ['ROBOFLOW_VERSION']}_summary.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Best model:", best_model_path)
print("Data yaml:", yaml_path)
print("Summary:", summary_path)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
