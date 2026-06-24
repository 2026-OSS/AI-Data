# YOLO11n v15 학습 결과 리포트

## 1. 작업 개요

본 문서는 그림책 객체 탐지를 위한 YOLO11n v15 모델의 학습 및 평가 결과를 정리한 리포트이다.

YOLO11n v15 모델은 Roboflow v15 통합 데이터셋을 기준으로 학습되었으며, 학습 완료 모델은 `AI-Data/artifacts/yolo11-v15/weights/best.pt`로 정리하였다. 본 문서는 v15 모델의 학습 조건, validation 및 test set 성능, 클래스별 성능 차이, 이전 YOLO11n v11 모델과의 비교 시 주의사항, 향후 개선 방향을 정리하는 것을 목적으로 한다.

## 2. 사용 데이터셋

| 항목              | 내용                                 |
| --------------- | ---------------------------------- |
| Dataset source  | Roboflow                           |
| Workspace       | `2026-oss`                         |
| Project         | `ai-picture-book-object-detection` |
| Dataset version | `v15`                              |
| Export format   | YOLOv8 format                      |
| Task type       | Object Detection                   |
| Image size      | 640                                |

Roboflow v15 데이터셋은 그림책 객체 탐지를 위한 최신 통합 데이터셋 기준으로 사용되었다. v15는 기존 v11 이후 갱신된 버전이므로, 이전 모델 결과와 비교할 때는 데이터셋 구성, train/valid/test split, 각 클래스별 instance 수가 달라졌을 수 있음을 고려해야 한다.

따라서 v15 모델의 성능은 동일한 v15 데이터셋에서 재학습한 모델끼리 비교하는 것이 가장 공정하다. v11 이전 모델과의 비교는 실험 진행 과정에서의 참고용으로 해석한다.

## 3. 탐지 클래스

YOLO11n v15 모델은 총 10개 클래스를 기준으로 학습되었다.

| Class ID | Class Name          |
| -------: | ------------------- |
|        0 | `book_flower`       |
|        1 | `book_flowerpot`    |
|        2 | `book_monkey`       |
|        3 | `book_stone`        |
|        4 | `braille`           |
|        5 | `tactile_flower`    |
|        6 | `tactile_flowerpot` |
|        7 | `tactile_monkey`    |
|        8 | `tactile_stone`     |
|        9 | `text`              |

## 4. 모델 정보

| 항목               | 내용                                                                 |
| ---------------- | ------------------------------------------------------------------ |
| Model            | YOLO11n                                                            |
| Initial weight   | `yolo11n.pt`                                                       |
| Dataset          | Roboflow v15                                                       |
| Run name         | `picture_book_yolo11n_v15`                                         |
| Epochs           | 100                                                                |
| Image size       | 640                                                                |
| Final model file | `artifacts/yolo11-v15/weights/best.pt`                             |
| Dataset yaml     | `artifacts/yolo11-v15/configs/data.yaml`                           |
| Notebook         | `notebooks/yolo11n_v15_training.ipynb`                             |
| Colab best path  | `/home/user/AI-Data/runs/detect/runs/detect/picture_book_yolo11n_v15/weights/best.pt` |
| Training env     | Ultralytics 8.4.71, Python 3.12.7, torch 2.11.0+cu128, CUDA H100   |

학습은 Ultralytics YOLO11n 모델을 기반으로 진행되었다. 노트북 실행 환경에서 확인된 모델 정보는 fused 기준 101 layers, 2,584,102 parameters, 6.3 GFLOPs이다.

## 5. 전체 성능 결과

test set 기준 전체 성능은 다음과 같다.

| Metric       | Score |
| ------------ | ----: |
| Precision    | 0.936 |
| Recall       | 0.842 |
| mAP@0.5      | 0.896 |
| mAP@0.5:0.95 | 0.689 |

v15 모델은 test set 기준 Precision 0.936, Recall 0.842를 기록하였다. Precision이 높아 모델이 탐지한 객체는 비교적 정확한 편이지만, Recall은 0.842로 일부 객체를 놓칠 가능성이 남아 있다. mAP@0.5는 0.896으로 객체를 대략적으로 찾는 능력은 안정적이며, mAP@0.5:0.95는 0.689로 bbox 정밀도 측면에서는 추가 검증이 필요하다.

본 프로젝트는 손끝으로 그림책 속 객체를 가리켰을 때 해당 객체를 탐지하고 설명을 제공하는 구조이므로, 실제 서비스 관점에서는 Recall과 bbox 정밀도가 모두 중요하다. 특히 손끝 좌표와 bbox 간 거리 계산을 사용하는 경우, 객체를 찾았더라도 bbox가 불안정하면 잘못된 객체가 선택될 수 있다.

## 6. Validation Set 성능

validation set 기준 전체 성능은 다음과 같다.

| Metric       | Score |
| ------------ | ----: |
| Images       |   247 |
| Instances    |   232 |
| Precision    | 0.929 |
| Recall       | 0.904 |
| mAP@0.5      | 0.947 |
| mAP@0.5:0.95 | 0.743 |

validation set에서는 Recall 0.904, mAP@0.5 0.947, mAP@0.5:0.95 0.743으로 test set보다 높은 성능을 보였다. 이는 validation set과 test set의 분포 차이, test set의 클래스별 instance 수, 촬영 조건 차이 등에 따른 결과일 수 있다. 최종 모델 판단 시에는 test set 성능을 우선 기준으로 보되, validation 결과와 실제 웹캠 테스트 결과를 함께 확인하는 것이 좋다.

## 7. 클래스별 성능 결과

test set 기준 클래스별 성능은 다음과 같다.

| Class             | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ----------------- | -----: | --------: | --------: | -----: | ------: | -----------: |
| all               |    123 |       117 |     0.936 |  0.842 |   0.896 |        0.689 |
| book_flower       |      7 |         7 |     0.967 |  0.714 |   0.897 |        0.506 |
| book_flowerpot    |     15 |        15 |     0.860 |  0.867 |   0.894 |        0.626 |
| book_monkey       |     12 |        12 |     0.870 |  0.750 |   0.881 |        0.685 |
| book_stone        |      6 |         6 |     1.000 |  0.729 |   0.835 |        0.725 |
| braille           |     16 |        16 |     0.838 |  0.974 |   0.838 |        0.708 |
| tactile_flower    |      2 |         2 |     0.985 |  1.000 |   0.995 |        0.796 |
| tactile_flowerpot |     17 |        17 |     0.871 |  0.797 |   0.914 |        0.713 |
| tactile_monkey    |      9 |         9 |     1.000 |  0.776 |   0.827 |        0.668 |
| tactile_stone     |      6 |         6 |     0.976 |  1.000 |   0.995 |        0.805 |
| text              |     27 |        27 |     0.990 |  0.815 |   0.885 |        0.660 |

## 8. 클래스별 성능 해석

### 8.1 상대적으로 안정적인 클래스

다음 클래스는 test set 기준으로 비교적 안정적인 성능을 보였다.

* `tactile_stone`
* `tactile_flower`
* `tactile_flowerpot`
* `braille`
* `book_stone`
* `text`

`tactile_stone`은 Precision 0.976, Recall 1.000, mAP@0.5:0.95 0.805로 가장 안정적인 축에 속한다. `tactile_flower`도 Recall 1.000, mAP@0.5 0.995, mAP@0.5:0.95 0.796으로 높은 성능을 보였지만, test instance가 2개뿐이므로 일반화 성능을 단정하기에는 표본 수가 부족하다.

`braille`은 Recall 0.974로 점자 영역을 놓치지 않는 성능이 좋게 나타났다. 다만 Precision은 0.838이므로 실제 입력에서 점자 영역 오탐 사례가 있는지 확인할 필요가 있다.

### 8.2 개선이 필요한 클래스

다음 클래스는 Recall 또는 mAP@0.5:0.95 기준으로 추가 확인이 필요하다.

* `book_flower`
* `book_monkey`
* `book_stone`
* `tactile_monkey`
* `text`

`book_flower`는 Precision 0.967, mAP@0.5 0.897로 탐지 자체는 비교적 정확하지만 Recall 0.714, mAP@0.5:0.95 0.506으로 나타났다. 즉, 일부 객체를 놓치거나 bbox가 정밀하게 맞지 않는 사례가 있을 수 있다.

`book_monkey`와 `book_stone`은 Precision은 높지만 Recall이 각각 0.750, 0.729로 낮은 편이다. 실제 서비스에서는 책 속 객체를 놓치면 설명 응답 자체가 발생하지 않을 수 있으므로, 해당 클래스의 미탐 사례를 우선 확인해야 한다.

`tactile_monkey`는 Precision 1.000으로 오탐은 적지만 Recall 0.776으로 일부 누락 가능성이 있다. `text` 역시 Precision 0.990으로 높지만 Recall 0.815, mAP@0.5:0.95 0.660이므로 실제 페이지 촬영 환경에서 bbox 안정성을 추가 검증해야 한다.

## 9. 이전 모델과의 비교

기존 YOLOv5s baseline, YOLO11n v6, YOLO11n v10, YOLO11n v11, YOLO11n v15 결과는 다음과 같다.

| Model   | Dataset      | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 비고            |
| ------- | ------------ | --------: | -----: | ------: | -----------: | ------------- |
| YOLOv5s | Roboflow v2  |     0.847 |  0.889 |   0.898 |        0.641 | 초기 baseline   |
| YOLO11n | Roboflow v6  |     0.835 |  0.719 |   0.776 |        0.545 | v6 통합 데이터셋 기준 |
| YOLO11n | Roboflow v10 |     0.894 |  0.781 |   0.831 |        0.647 | v10 통합 데이터셋 기준 |
| YOLO11n | Roboflow v11 |     0.903 |  0.857 |   0.919 |        0.708 | v11 통합 데이터셋 기준 |
| YOLO11n | Roboflow v15 |     0.936 |  0.842 |   0.896 |        0.689 | v15 통합 데이터셋 기준 |

수치상으로 v15는 v11 대비 Precision이 상승했지만, Recall, mAP@0.5, mAP@0.5:0.95는 낮게 나타났다. 다만 위 비교는 데이터셋 버전이 서로 다르므로 모델 자체의 절대적인 우열로 해석하기에는 제한이 있다. v15 데이터셋에서 최종 모델을 선정하려면 YOLO11n v11과 동일한 조건으로 비교하는 대신, 동일한 Roboflow v15 데이터셋에서 후보 모델을 재학습하고 평가해야 한다.

## 10. 현재 모델의 장점과 한계

### 10.1 장점

* YOLO11n 기반의 가벼운 객체 탐지 모델을 Roboflow v15 데이터셋 기준으로 확보하였다.
* test set 기준 Precision 0.936으로 오탐을 줄이는 방향의 성능이 좋게 나타났다.
* `tactile_stone`, `tactile_flower`, `tactile_flowerpot` 등 촉각 객체류 일부 클래스에서 높은 mAP@0.5:0.95를 기록했다.
* `braille`의 Recall이 0.974로 높아 점자 영역 누락 가능성이 낮게 나타났다.
* 모델 산출물과 그래프를 `artifacts/yolo11-v15` 아래에 정리하여 백엔드 및 웹캠 테스트에서 활용할 수 있도록 했다.

### 10.2 한계

* test set 기준 Recall 0.842로 일부 객체 누락 가능성이 있다.
* `book_flower`, `book_monkey`, `book_stone`의 Recall이 상대적으로 낮아 책 이미지 객체류의 미탐 사례 확인이 필요하다.
* `book_flower`의 mAP@0.5:0.95가 0.506으로 낮아 bbox 정밀도 개선 여부를 확인해야 한다.
* `tactile_flower`는 성능 수치가 높지만 test instance가 2개뿐이므로 표본 수에 따른 변동성이 크다.
* 학습 로그에서 segment와 box 수가 일치하지 않는다는 경고가 있었으며, 평가 시 box 기준으로만 사용되었다.
* 실제 웹캠 환경에서는 조명, 각도, 손가락 가림, 페이지 기울어짐 등의 영향으로 notebook test set 성능보다 낮아질 수 있다.

## 11. 백엔드 및 웹캠 연동 관점

v15 모델의 입력과 출력 형태는 다음과 같이 정리할 수 있다.

| 항목              | 내용                                |
| ----------------- | ----------------------------------- |
| Input             | image 또는 webcam frame             |
| Image size        | 640                                 |
| Output class      | detected object class name          |
| Output confidence | detection confidence score          |
| Output bbox       | `[x1, y1, x2, y2]`                  |
| Model path        | `artifacts/yolo11-v15/weights/best.pt` |

웹캠 테스트는 다음 명령으로 실행할 수 있다.

```bash
cd AI-Data
bash scripts/run_yolo11_webcam.sh
```

손끝으로 가리킨 객체를 선택하는 테스트는 다음 명령으로 실행한다.

```bash
cd AI-Data
HAND_MODE=point bash scripts/run_yolo11_webcam.sh
```

손끝 지시 방식에서는 단순 탐지 성능뿐 아니라 bbox 중심점, 손끝 좌표와 bbox 간 거리, confidence threshold가 실제 사용성에 큰 영향을 준다. v15는 Precision이 높으므로 오탐 억제에는 유리할 수 있지만, Recall이 v11보다 낮게 나타났으므로 손끝으로 가리킨 객체가 누락되는지 실제 촬영 환경에서 확인해야 한다.

## 12. 향후 개선 방향

현재 YOLO11n v15 모델은 최신 v15 데이터셋 기준의 기능 검증용 후보 모델로 사용할 수 있다. 다만 최종 적용 모델로 확정하기 전 다음 작업을 검토할 필요가 있다.

1. 실제 웹캠 입력 환경에서 손끝 지시 탐지 품질 확인
2. `book_flower`, `book_monkey`, `book_stone` 중심 미탐 사례 확인
3. `book_flower` bbox 정밀도 개선 여부 확인
4. confidence threshold별 탐지 결과 비교
5. 동일 Roboflow v15 데이터셋 기준으로 YOLO11s 또는 YOLO11m 추가 학습
6. v15 학습 결과를 JSON, CSV 형태로 `artifacts/yolo11-v15/results/` 아래에 추가 정리
7. test set 외 실제 촬영 이미지로 별도 검증 세트 구성
8. segment annotation이 섞인 데이터셋 항목을 점검하여 box annotation 기준으로 정리

최종 모델 선정 시에는 단순히 mAP만 보는 것이 아니라 다음 기준을 함께 고려해야 한다.

| 기준             | 이유                                       |
| -------------- | ---------------------------------------- |
| Precision      | 잘못된 객체를 탐지하는 오탐을 줄이기 위함                  |
| Recall         | 실제 객체를 놓치는 미탐을 줄이기 위함                    |
| mAP@0.5        | 객체를 대략적으로 잘 잡는지 확인하기 위함                  |
| mAP@0.5:0.95   | bbox가 정확하게 객체 영역을 감싸는지 확인하기 위함           |
| 모델 크기          | 백엔드 배포 및 로딩 부담을 줄이기 위함                   |
| 추론 속도          | 웹캠 기반 실시간 서비스 적용 가능성을 판단하기 위함            |
| 손끝 선택 정확도      | 탐지 bbox와 손끝 좌표가 실제 서비스 결과에 직접 연결되기 때문 |
| 약한 클래스 개선 여부   | 실제 서비스에서 특정 객체만 계속 누락되는 문제를 방지하기 위함   |

## 13. 최종 정리

YOLO11n v15 모델은 Roboflow v15 통합 데이터셋 기준으로 학습 및 평가되었으며, test set 기준 Precision 0.936, Recall 0.842, mAP@0.5 0.896, mAP@0.5:0.95 0.689의 성능을 보였다.

현재 결과만 보면 v15 모델은 Precision이 높아 오탐을 줄이는 방향에서는 강점이 있지만, v11과 비교했을 때 Recall과 bbox 정밀도 지표는 다소 낮게 나타났다. 따라서 최종 적용 여부를 판단하려면 동일한 v15 데이터셋 기준의 추가 후보 모델 비교와 실제 웹캠 기반 정성 평가가 필요하다.

특히 `book_flower`, `book_monkey`, `book_stone`의 미탐 사례, `book_flower`의 bbox 정밀도, 손끝 좌표 기반 객체 선택 품질은 우선적으로 확인하는 것이 좋다.

## 14. 관련 파일

* `artifacts/yolo11-v15/weights/best.pt`
* `artifacts/yolo11-v15/configs/data.yaml`
* `artifacts/yolo11-v15/results/graphs/confusion_matrix.png`
* `artifacts/yolo11-v15/results/graphs/results.png`
* `notebooks/yolo11n_v15_training.ipynb`
* `scripts/run_yolo11_webcam.sh`

## 15. 관련 이슈

* Related to #55
