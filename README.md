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

## Webcam YOLO11 Object Detector

`artifacts/yolo11-v11/best.pt` 모델로 웹캠 객체 탐지를 실행할 수 있습니다.

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
SOURCE=1 HAND_MODE=point POINT_MARGIN=40 bash scripts/run_yolo11_webcam.sh
```
