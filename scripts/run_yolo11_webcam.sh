#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/artifacts/yolo11-v15}"
CAMERA="${CAMERA:-builtin}"
PHONE_SOURCE="${PHONE_SOURCE:-1}"
PHONE_URL="${PHONE_URL:-}"
SOURCE="${SOURCE:-}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
IMG_SIZE="${IMG_SIZE:-960}"
CONF="${CONF:-0.15}"
DEVICE="${DEVICE:-}"
INFER_STRIDE="${INFER_STRIDE:-1}"
HAND_MODE="${HAND_MODE:-point}"
POINT_HOLD_SECONDS="${POINT_HOLD_SECONDS:-3.0}"
POINT_RESET_SECONDS="${POINT_RESET_SECONDS:-0.7}"
POINT_MARGIN="${POINT_MARGIN:-50}"
HAND_CONF="${HAND_CONF:-0.4}"
HAND_TRACK_CONF="${HAND_TRACK_CONF:-0.4}"
HAND_MODEL_PATH="${HAND_MODEL_PATH:-$ROOT_DIR/artifacts/hand-landmarker/hand_landmarker.task}"

if [[ -d "$MODEL_PATH" ]]; then
  if [[ -f "$MODEL_PATH/best.pt" ]]; then
    MODEL_PATH="$MODEL_PATH/best.pt"
  elif [[ -f "$MODEL_PATH/weights/best.pt" ]]; then
    MODEL_PATH="$MODEL_PATH/weights/best.pt"
  fi
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model file not found: $MODEL_PATH"
  echo "Set MODEL_PATH=/path/to/yolo11_best.pt or MODEL_PATH=/path/to/artifact-dir."
  exit 1
fi

if [[ -z "$SOURCE" ]]; then
  case "$CAMERA" in
    builtin)
      SOURCE="0"
      ;;
    phone)
      SOURCE="$PHONE_SOURCE"
      ;;
    url)
      if [[ -z "$PHONE_URL" ]]; then
        echo "PHONE_URL is required when CAMERA=url"
        echo "Example: CAMERA=url PHONE_URL=http://192.168.0.10:4747/video $0"
        exit 1
      fi
      SOURCE="$PHONE_URL"
      ;;
    *)
      echo "Unknown CAMERA value: $CAMERA"
      echo "Use CAMERA=builtin, CAMERA=phone, CAMERA=url, or set SOURCE directly."
      exit 1
      ;;
  esac
fi

mkdir -p "$ROOT_DIR/.cache/matplotlib" "$ROOT_DIR/.cache/ultralytics"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ROOT_DIR/.cache/ultralytics}"

echo "Using camera source: $SOURCE"
echo "Using model: $MODEL_PATH"

if command -v conda >/dev/null 2>&1 && [[ -n "$CONDA_ENV" ]]; then
  conda run -n "$CONDA_ENV" python "$ROOT_DIR/scripts/yolo11_webcam.py" \
    --model "$MODEL_PATH" \
    --source "$SOURCE" \
    --imgsz "$IMG_SIZE" \
    --conf "$CONF" \
    --device "$DEVICE" \
    --infer-stride "$INFER_STRIDE" \
    --hand-mode "$HAND_MODE" \
    --point-hold-seconds "$POINT_HOLD_SECONDS" \
    --point-reset-seconds "$POINT_RESET_SECONDS" \
    --point-margin "$POINT_MARGIN" \
    --hand-conf "$HAND_CONF" \
    --hand-track-conf "$HAND_TRACK_CONF" \
    --hand-model "$HAND_MODEL_PATH"
else
  "$PYTHON_BIN" "$ROOT_DIR/scripts/yolo11_webcam.py" \
    --model "$MODEL_PATH" \
    --source "$SOURCE" \
    --imgsz "$IMG_SIZE" \
    --conf "$CONF" \
    --device "$DEVICE" \
    --infer-stride "$INFER_STRIDE" \
    --hand-mode "$HAND_MODE" \
    --point-hold-seconds "$POINT_HOLD_SECONDS" \
    --point-reset-seconds "$POINT_RESET_SECONDS" \
    --point-margin "$POINT_MARGIN" \
    --hand-conf "$HAND_CONF" \
    --hand-track-conf "$HAND_TRACK_CONF" \
    --hand-model "$HAND_MODEL_PATH"
fi
