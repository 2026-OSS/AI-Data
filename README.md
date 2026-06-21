# AI-Data

촉각형 그림책 기반 객체 인식 프로젝트의 데이터셋 관리 레포지토리입니다.

## Repository Structure

- docs : 데이터셋 구조 및 클래스 정의 문서
- dataset : 학습 데이터셋 (추후 추가 예정)
- notebooks : 모델 학습 및 실험 노트북
- artifacts : 학습된 모델 및 백엔드 전달용 산출물

## Documents

- [01_dataset_structure.md](docs/01_dataset_structure.md): 데이터셋 구조 및 클래스 정의
- [02_dataset_collection.md](docs/02_dataset_collection.md): 객체 탐지 데이터셋 수집 기준
- [03_dataset_labeling.md](docs/03_dataset_labeling.md): Roboflow 데이터 라벨링 및 페이지별 음성 설명 매핑
- [04_dataset_preprocessing_augmentation.md](docs/04_dataset_preprocessing_augmentation.md): 데이터 전처리, 증강, 분할 검수 및 Roboflow Version 생성 기준
- [06_yolov5_training_environment.md](docs/06_yolov5_training_environment.md): Google Colab 기반 YOLOv5·YOLOv11 학습 환경 구축 및 검증 가이드

## Webcam Page Classifier

`artifacts/page-classifier-mobilenetv2/page_classifier_mobilenetv2.keras` 모델로 웹캠 페이지 분류를 실행할 수 있습니다.

```bash
cd AI-Data
bash scripts/run_page_classifier_webcam.sh
```

기본 클래스 순서는 훈련 노트북 기준 `none,page1,page2,page3`입니다. 다른 모델이나 카메라를 쓸 때는 환경변수로 바꿀 수 있습니다.

```bash
MODEL_PATH=artifacts/page-classifier-mobilenetv2/page_classifier_mobilenetv2.keras SOURCE=1 bash scripts/run_page_classifier_webcam.sh
```

웹캠 페이지 분류는 최근 프레임 EMA smoothing, `top1 - top2` margin 검사, 연속 reliable frame 확인을 함께 사용합니다. 페이지가 자주 튀면 `SMOOTHING_ALPHA`를 낮추거나 `STABLE_FRAMES`를 올리고, 애매한 화면에서 label이 너무 쉽게 뜨면 `MARGIN_THRESHOLD`, `THRESHOLD`, `NONE_THRESHOLD`를 올립니다.

```bash
THRESHOLD=0.65 MARGIN_THRESHOLD=0.20 SMOOTHING_ALPHA=0.25 STABLE_FRAMES=3 bash scripts/run_page_classifier_webcam.sh
```

산출물에 포함된 class names JSON을 명시해서 실행할 수도 있습니다.

```bash
LABELS_FILE=artifacts/page-classifier-mobilenetv2/page_classifier_class_names.json bash scripts/run_page_classifier_webcam.sh
```

## Webcam YOLO11 Object Detector

`artifacts/yolo11-v15/weights/best.pt` 모델로 웹캠 객체 탐지를 실행할 수 있습니다.

```bash
cd AI-Data
bash scripts/run_yolo11_webcam.sh
```

기본 카메라는 `SOURCE=0`입니다. 다른 카메라나 iPhone Continuity Camera가 잡히면 `SOURCE=1`처럼 바꿔 실행합니다.

```bash
SOURCE=1 bash scripts/run_yolo11_webcam.sh
```

손끝으로 객체를 가리키고 3초간 유지해 선택하는 모드는 다음처럼 켭니다.

```bash
HAND_MODE=point bash scripts/run_yolo11_webcam.sh
```

이 모드에서는 손끝이 가장 많이 가리킨 객체 하나만 화면에 표시됩니다. 손끝 판정 범위가 너무 빡빡하거나 넓으면 `POINT_MARGIN` 값을 조절합니다.

```bash
SOURCE=1 HAND_MODE=point POINT_MARGIN=60 bash scripts/run_yolo11_webcam.sh
```

객체가 잘 안 잡히면 confidence를 낮추거나 입력 크기를 키워 작은 객체 감도를 올릴 수 있습니다. 기본값은 `CONF=0.15`, `IMG_SIZE=960`, `POINT_MARGIN=50`입니다.

```bash
SOURCE=1 CONF=0.10 IMG_SIZE=960 POINT_MARGIN=60 bash scripts/run_yolo11_webcam.sh
```

## Predict API Server

백엔드 연동용 추론 서버는 기본으로 현재 안정 버전 모델을 사용합니다.

- YOLO 모델: `artifacts/yolo11-v15/weights/best.pt`
- YOLO 클래스 정보: `artifacts/yolo11-v15/configs/data.yaml`
- 페이지 분류 모델: `artifacts/page-classifier-mobilenetv2/page_classifier_mobilenetv2.keras`
- 손끝 추출 모델: `artifacts/hand-landmarker/hand_landmarker.task`

```bash
python3 -m uvicorn server.main:app --host 127.0.0.1 --port 8001
```

기본 모델 경로는 프로젝트 루트 기준으로 해석되므로, 다른 폴더에서 실행해도 기본 산출물은 동일하게 찾습니다.
환경변수로 상대경로를 지정할 때도 `AI-Data` 루트 기준 경로를 사용합니다.

서버 추론도 기본적으로 웹캠 테스트와 같은 감도 설정을 사용합니다.
`/predict`는 YOLO 객체 검출 결과 전체와 손끝 좌표를 반환하고, 손끝-객체 매칭은 백엔드 `/api/interaction/detect`에서 처리합니다.

- `YOLO_CONF=0.15`: YOLO confidence threshold
- `YOLO_IMGSZ=960`: YOLO inference image size
- `HAND_MIN_DETECTION_CONFIDENCE=0.4`: 손 검출 confidence
- `HAND_MIN_PRESENCE_CONFIDENCE=0.4`: 손 존재 confidence
- `HAND_MIN_TRACKING_CONFIDENCE=0.4`: 손 추적 confidence
- `PAGE_CONFIDENCE_THRESHOLD=0.75`: 페이지 label을 신뢰할 최소 confidence
- `PAGE_MARGIN_THRESHOLD=0.15`: 페이지 `top1 - top2` 최소 margin
- `PAGE_SMOOTHING_ALPHA=0.35`: 최신 페이지 예측을 EMA에 반영하는 비율
- `PAGE_STABLE_FRAMES=2`: 같은 reliable page가 연속으로 나와야 확정하는 최소 frame 수
- `PAGE_NONE_CONFIDENCE_THRESHOLD`: `none` 클래스에만 적용할 별도 confidence threshold
- `PAGE_CLASSES_FILE`: JSON 또는 line-based text class names 파일 경로

오탐이 늘면 `YOLO_CONF`를 `0.20~0.25`로 올리고, 작은 객체를 더 잡아야 하면 `YOLO_CONF=0.10` 또는 `YOLO_IMGSZ=1280`을 시도합니다.

페이지 예측이 불안정하면 AI 서버는 `/predict` 응답의 `page`에 `top_k`, `margin`, `raw`, `smoothed`, `reliable`을 함께 넣습니다. `reliable=false`이면 마지막으로 확정된 label을 유지하되 confidence를 threshold 아래로 낮춰 BE fallback이 작동하게 합니다.

YOLO 모델 버전이 바뀌면 코드 수정 없이 실행할 때 경로만 바꿉니다.

```bash
YOLO_MODEL_PATH=artifacts/yolo11-v15/weights/best.pt \
YOLO_DATA_YAML=artifacts/yolo11-v15/configs/data.yaml \
python3 -m uvicorn server.main:app --host 127.0.0.1 --port 8001
```

페이지 모델이나 손끝 추출 모델을 바꿀 때도 같은 방식으로 환경변수를 지정합니다.

```bash
PAGE_MODEL_PATH=artifacts/page-classifier-mobilenetv2/page_classifier_mobilenetv2.keras \
HAND_LANDMARKER_MODEL_PATH=artifacts/hand-landmarker/hand_landmarker.task \
python3 -m uvicorn server.main:app --host 127.0.0.1 --port 8001
```
