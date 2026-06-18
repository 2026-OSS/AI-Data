# YOLOv5 and YOLOv11 Training Environment Guide

## 개요

본 문서는 YOLOv5 및 YOLOv11 기반 객체 탐지 모델 학습 환경 구축 기준을 정의한다.

대상 이슈는 `#6 [FEAT] YOLOv5·YOLOv11 학습 환경 구축`이다.

Google Colab 환경에서 YOLOv5와 YOLOv11을 설치하고, Roboflow 데이터셋을 연동하여 학습 가능한 상태를 구성한다. 또한 데이터셋 다운로드, `data.yaml` 설정, GPU 환경 확인, 테스트 학습, 결과 파일 생성, 추론 실행 여부를 확인하여 학습 환경이 정상 동작하는지 검증한다.

이 문서의 목적은 최종 성능 비교가 아니라, 두 모델 모두 같은 데이터셋 기준으로 학습을 시작할 수 있는 Colab 환경을 재현 가능하게 정리하는 것이다.

---

## 사용 환경

| 항목 | 기준 |
| --- | --- |
| 실행 환경 | Google Colab |
| Python | Python 3.x |
| 딥러닝 프레임워크 | PyTorch |
| 학습 모델 | YOLOv5, YOLOv11 |
| YOLOv11 패키지 | Ultralytics |
| 데이터셋 관리 | Roboflow |
| 데이터셋 이름 | `AI-Picture-Book-Object-Detection` |
| Export 형식 | YOLOv5: `yolov5`, YOLOv11: `yolov8` |

Colab에서 실행할 때는 `Runtime > Change runtime type > Hardware accelerator > GPU`를 선택한다.

---

## 산출물

| 산출물 | 위치 |
| --- | --- |
| 학습 환경 구축 가이드 | `docs/06_yolov5_training_environment.md` |
| YOLOv5 학습 코드 | `scripts/train_yolov5_gpu.sh` |
| YOLOv5 Colab Notebook | `notebooks/yolov5_baseline_training_v6.ipynb` |
| YOLOv5 학습 결과 | `runs/train/<run_name>/weights/best.pt` |
| YOLOv11 학습 결과 | `runs/detect/<run_name>/weights/best.pt` |
| 백엔드 전달용 모델 | `artifacts/yolov5-v*/best.pt` 또는 별도 YOLOv11 산출물 |

테스트 학습 산출물은 Colab 런타임 안에서 먼저 확인하고, 최종 모델로 사용할 `best.pt`와 `data.yaml`만 별도 artifact 경로에 정리한다.

---

## 공통 Colab 설정

### 0. 공통 변수 설정

노트북 상단에서 Roboflow 프로젝트와 학습 설정을 먼저 고정한다.

```python
WORKSPACE = "2026-oss"
PROJECT = "ai-picture-book-object-detection"
ROBOFLOW_VERSION = 6

IMG_SIZE = 640
BATCH_SIZE = 16
TEST_EPOCHS = 3
TRAIN_EPOCHS = 100
```

`ROBOFLOW_VERSION`은 실제 학습에 사용할 Roboflow Version에 맞춰 변경한다.

### 1. GPU 환경 확인

```python
!nvidia-smi

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
```

확인 기준은 다음과 같다.

- `nvidia-smi`에서 GPU 정보가 출력되어야 한다.
- `torch.cuda.is_available()` 값이 `True`여야 한다.
- GPU 이름이 출력되어야 한다.

### 2. Roboflow API Key 설정

Roboflow API Key는 노트북에 직접 하드코딩하지 않는다.

```python
import os
from getpass import getpass

os.environ["ROBOFLOW_API_KEY"] = getpass("Roboflow API Key: ")
```

### 3. Roboflow 데이터셋 다운로드

YOLOv5는 Roboflow `yolov5` export를 사용한다. YOLOv11은 Ultralytics 계열이므로 Roboflow `yolov8` export를 사용하면 `data.yaml`을 그대로 연결하기 쉽다.

```python
!pip install -q roboflow

import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace(WORKSPACE).project(PROJECT)

# YOLOv5 학습용
dataset = project.version(ROBOFLOW_VERSION).download("yolov5")

print("Dataset Location:", dataset.location)
```

YOLOv11 학습만 별도로 진행하는 경우에는 다음처럼 다운로드한다.

```python
dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
print("Dataset Location:", dataset.location)
```

### 4. data.yaml 확인

```python
from pathlib import Path

dataset_dir = Path(dataset.location)
yaml_path = dataset_dir / "data.yaml"

print(yaml_path)
print(yaml_path.read_text())
```

`data.yaml`에서 다음 항목을 확인한다.

- `train`, `val`, `test` 경로가 실제 디렉터리와 일치하는지 확인한다.
- `nc`가 클래스 수와 일치하는지 확인한다.
- `names` 클래스 순서가 프로젝트 기준과 일치하는지 확인한다.

### 5. 데이터 경로 확인

```python
!ls {dataset.location}
!ls {dataset.location}/train/images | head
!ls {dataset.location}/train/labels | head
!ls {dataset.location}/valid/images | head
!ls {dataset.location}/test/images | head
```

다음 구조가 있어야 한다.

```text
AI-Picture-Book-Object-Detection-*/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## YOLOv5 학습 환경

### 1. YOLOv5 설치

```python
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt
!pip install -q roboflow
```

### 2. YOLOv5용 data.yaml 경로 보정

YOLOv5를 `/content/yolov5`에서 실행하는 경우, Roboflow 다운로드 위치에 따라 `data.yaml` 경로를 보정한다. 공통 다운로드 셀을 `/content`에서 먼저 실행했다면, `%cd yolov5` 이후 YOLOv5용 다운로드 셀을 한 번 더 실행하거나 데이터셋 폴더를 `/content/yolov5` 아래로 옮긴다.

```python
from pathlib import Path

dataset_dir = Path(f"./AI-Picture-Book-Object-Detection-{ROBOFLOW_VERSION}")
yaml_path = dataset_dir / "data.yaml"
data = yaml_path.read_text()

replacements = {
    "train: train/images": f"train: ./{dataset_dir.name}/train/images",
    "val: valid/images": f"val: ./{dataset_dir.name}/valid/images",
    "test: test/images": f"test: ./{dataset_dir.name}/test/images",
    "train: ../train/images": f"train: ./{dataset_dir.name}/train/images",
    "val: ../valid/images": f"val: ./{dataset_dir.name}/valid/images",
    "test: ../test/images": f"test: ./{dataset_dir.name}/test/images",
}

for old, new in replacements.items():
    data = data.replace(old, new)

yaml_path.write_text(data)
print(yaml_path.read_text())
```

### 3. YOLOv5 테스트 학습

학습 환경이 정상 동작하는지 빠르게 확인하기 위해 짧은 epoch으로 테스트 학습을 수행한다.

```python
!python train.py \
  --img $IMG_SIZE \
  --batch $BATCH_SIZE \
  --epochs $TEST_EPOCHS \
  --data ./AI-Picture-Book-Object-Detection-{ROBOFLOW_VERSION}/data.yaml \
  --weights yolov5s.pt \
  --device 0 \
  --name picture_book_yolov5_test
```

정식 학습에서는 epoch 수와 run name을 조정한다.

```python
!python train.py \
  --img $IMG_SIZE \
  --batch $BATCH_SIZE \
  --epochs $TRAIN_EPOCHS \
  --data ./AI-Picture-Book-Object-Detection-{ROBOFLOW_VERSION}/data.yaml \
  --weights yolov5s.pt \
  --device 0 \
  --name picture_book_yolov5_v{ROBOFLOW_VERSION}
```

### 4. YOLOv5 결과 및 추론 확인

```python
!ls runs/train/picture_book_yolov5_test
!ls runs/train/picture_book_yolov5_test/weights
```

확인할 주요 파일은 다음과 같다.

```text
runs/train/picture_book_yolov5_test/
├── results.csv
├── results.png
├── confusion_matrix.png
└── weights/
    ├── best.pt
    └── last.pt
```

학습된 `best.pt`로 test 이미지를 추론한다.

```python
!python detect.py \
  --weights runs/train/picture_book_yolov5_test/weights/best.pt \
  --source AI-Picture-Book-Object-Detection-{ROBOFLOW_VERSION}/test/images \
  --img $IMG_SIZE \
  --conf 0.25 \
  --name picture_book_yolov5_test_infer
```

추론 결과는 `runs/detect/picture_book_yolov5_test_infer/`에서 확인한다.

---

## YOLOv11 학습 환경

### 1. YOLOv11 설치

YOLOv11은 Ultralytics 패키지를 사용한다.

```python
!pip install -q ultralytics roboflow

from ultralytics import YOLO

YOLO("yolo11n.pt")
```

### 2. YOLOv11 테스트 학습

Roboflow에서 다운로드한 `data.yaml`을 그대로 사용하여 테스트 학습을 수행한다.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=TEST_EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=0,
    name="picture_book_yolo11_test",
)
```

Colab 셸 명령으로 실행할 수도 있다.

```python
!yolo detect train \
  model=yolo11n.pt \
  data={dataset.location}/data.yaml \
  epochs={TEST_EPOCHS} \
  imgsz={IMG_SIZE} \
  batch={BATCH_SIZE} \
  device=0 \
  name=picture_book_yolo11_test
```

정식 학습에서는 epoch 수와 run name을 조정한다.

```python
!yolo detect train \
  model=yolo11n.pt \
  data={dataset.location}/data.yaml \
  epochs={TRAIN_EPOCHS} \
  imgsz={IMG_SIZE} \
  batch={BATCH_SIZE} \
  device=0 \
  name=picture_book_yolo11_v{ROBOFLOW_VERSION}
```

### 3. YOLOv11 결과 및 추론 확인

학습 완료 후 결과 파일을 확인한다.

```python
!ls runs/detect/picture_book_yolo11_test
!ls runs/detect/picture_book_yolo11_test/weights
```

확인할 주요 파일은 다음과 같다.

```text
runs/detect/picture_book_yolo11_test/
├── results.csv
├── results.png
├── confusion_matrix.png
└── weights/
    ├── best.pt
    └── last.pt
```

학습된 `best.pt`로 test 이미지를 추론한다.

```python
!yolo detect predict \
  model=runs/detect/picture_book_yolo11_test/weights/best.pt \
  source={dataset.location}/test/images \
  imgsz={IMG_SIZE} \
  conf=0.25 \
  name=picture_book_yolo11_test_infer
```

추론 결과는 `runs/detect/picture_book_yolo11_test_infer/`에서 확인한다.

---

## 로컬/Colab 스크립트 실행

반복 실행이 필요한 경우 YOLOv5는 `scripts/train_yolov5_gpu.sh`를 사용할 수 있다.

```bash
cd AI-Data
export ROBOFLOW_API_KEY="your_key_here"

EPOCHS=3 \
BATCH_SIZE=16 \
IMG_SIZE=640 \
RUN_NAME=picture_book_yolov5_test \
ROBOFLOW_VERSION=6 \
bash scripts/train_yolov5_gpu.sh
```

이 스크립트는 다음 작업을 수행한다.

- YOLOv5 저장소 확인 및 clone
- 필수 라이브러리 설치
- Roboflow 데이터셋 다운로드
- `data.yaml` 경로 수정
- GPU 상태 확인
- YOLOv5 학습 실행

YOLOv11은 Colab에서 Ultralytics CLI 또는 Python API로 실행한다.

```bash
yolo detect train model=yolo11n.pt data=/path/to/data.yaml epochs="${EPOCHS:-3}" imgsz="${IMG_SIZE:-640}" batch="${BATCH_SIZE:-16}" device=0
```

---

## 검증 기준

| 검증 항목 | YOLOv5 기준 | YOLOv11 기준 |
| --- | --- | --- |
| GPU 정상 인식 | `nvidia-smi`, `torch.cuda.is_available()` 확인 | `nvidia-smi`, `torch.cuda.is_available()` 확인 |
| 데이터셋 로딩 | `train.py`에서 `data.yaml` 로딩 | `yolo detect train`에서 `data.yaml` 로딩 |
| `data.yaml` 설정 | `nc`, `names`, `train`, `val`, `test` 경로 확인 | `nc`, `names`, `train`, `val`, `test` 경로 확인 |
| 학습 시작 가능 여부 | epoch 로그 출력 | epoch 로그 출력 |
| 결과 파일 생성 | `runs/train/<run_name>/weights/best.pt` 생성 | `runs/detect/<run_name>/weights/best.pt` 생성 |
| 추론 실행 가능 여부 | `detect.py` 실행 결과 생성 | `yolo detect predict` 실행 결과 생성 |

환경 구축 완료 여부는 테스트 epoch 기준으로 판단한다. 성능 평가는 별도 학습 리포트에서 정리하며, 이 문서에서는 학습 시작과 산출물 생성이 정상적으로 되는지를 우선 확인한다.

---

## 문제 해결

### GPU가 인식되지 않는 경우

- Colab 메뉴에서 `Runtime > Change runtime type > GPU`가 선택되어 있는지 확인한다.
- 런타임을 재시작한 뒤 `nvidia-smi`를 다시 실행한다.
- 무료 Colab 사용량 제한으로 GPU가 배정되지 않을 수 있으므로 일정 시간 후 다시 시도한다.

### Roboflow 다운로드가 실패하는 경우

- `ROBOFLOW_API_KEY`가 올바르게 설정되었는지 확인한다.
- workspace 이름이 `2026-oss`인지 확인한다.
- project 이름이 `ai-picture-book-object-detection`인지 확인한다.
- 사용할 Version 번호가 Roboflow에 실제로 존재하는지 확인한다.

### data.yaml 경로 오류가 발생하는 경우

- `train`, `val`, `test` 경로가 현재 실행 위치 기준에서 존재하는지 확인한다.
- `!ls <dataset_dir>/train/images` 명령으로 실제 이미지 경로를 확인한다.
- YOLOv5는 `/content/yolov5` 기준 상대 경로를 사용하므로 필요하면 경로를 보정한다.
- YOLOv11은 절대 경로 또는 `dataset.location` 기반 경로를 사용하면 경로 오류를 줄일 수 있다.

### best.pt가 생성되지 않는 경우

- 학습이 중간에 중단되었는지 로그를 확인한다.
- batch size가 GPU 메모리에 비해 너무 큰 경우 `--batch` 또는 `batch` 값을 줄인다.
- YOLOv5는 `runs/train/<run_name>/weights/`를 확인한다.
- YOLOv11은 `runs/detect/<run_name>/weights/`를 확인한다.

---

## 체크리스트

### 학습 환경 구축

- [ ] Google Colab 환경 구성
- [ ] YOLOv5 설치
- [ ] YOLOv11 설치
- [ ] 필수 라이브러리 설치
- [ ] GPU 환경 확인

### 데이터셋 연동

- [ ] Roboflow API Key 설정
- [ ] Roboflow 데이터셋 다운로드
- [ ] `data.yaml` 확인
- [ ] 데이터 경로 확인

### 학습 환경 검증

- [ ] YOLOv5 테스트 학습 수행
- [ ] YOLOv11 테스트 학습 수행
- [ ] 학습 로그 확인
- [ ] 결과 파일 생성 확인
- [ ] 추론 가능 여부 확인

### 최종 정리

- [ ] Colab Notebook에서 전체 셀 순차 실행 가능
- [ ] YOLOv5 `best.pt` 생성 경로 확인
- [ ] YOLOv11 `best.pt` 생성 경로 확인
- [ ] 테스트 이미지 추론 결과 폴더 확인
- [ ] 최종 산출물 경로를 README 또는 학습 리포트에 기록

---

## 관련 문서

- [01_dataset_structure.md](01_dataset_structure.md): 데이터셋 구조 및 클래스 정의
- [03_dataset_labeling.md](03_dataset_labeling.md): Roboflow 데이터 라벨링 및 페이지별 음성 설명 매핑
- [04_dataset_preprocessing_augmentation.md](04_dataset_preprocessing_augmentation.md): 데이터 전처리, 증강, 분할 검수 및 Roboflow Version 생성 기준
