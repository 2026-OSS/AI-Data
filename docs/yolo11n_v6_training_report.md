# YOLO11n v6 학습 결과 리포트

## 1. 작업 개요

본 문서는 그림책 객체 탐지를 위한 YOLO11n v6 모델의 학습 및 평가 결과를 정리한 리포트이다.

YOLO11n v6 모델은 Roboflow v6 통합 데이터셋을 기준으로 학습되었으며, 학습 완료 모델 파일과 평가 결과 파일은 AI 레포에서 관리한다. 본 문서는 모델 성능, 클래스별 성능 차이, YOLOv5s baseline과의 비교 시 주의사항, 향후 개선 방향을 정리하는 것을 목적으로 한다.

## 2. 사용 데이터셋

| 항목              | 내용                                 |
| --------------- | ---------------------------------- |
| Dataset source  | Roboflow                           |
| Workspace       | `2026-oss`                         |
| Project         | `ai-picture-book-object-detection` |
| Dataset version | `v6`                               |
| Export format   | YOLOv8 format                      |
| Task type       | Object Detection                   |
| Image size      | 640                                |

Roboflow v6 데이터셋은 최종 통합 데이터셋 기준으로 사용되었다. 기존 YOLOv5s baseline은 Roboflow v2 데이터셋을 기준으로 학습 및 평가되었으므로, YOLOv5s baseline 결과와 YOLO11n v6 결과는 동일 조건의 직접 비교가 아니다.

따라서 두 모델의 성능 비교는 모델 자체의 우열 판단보다는, 실험 진행 과정에서의 참고용 비교로 해석해야 한다.

## 3. 탐지 클래스

YOLO11n v6 모델은 총 10개 클래스를 기준으로 학습되었다.

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

| 항목                | 내용                                     |
| ----------------- | -------------------------------------- |
| Model             | YOLO11n                                |
| Initial weight    | `yolo11n.pt`                           |
| Dataset           | Roboflow v6                            |
| Epochs            | 100                                    |
| Batch size        | 16                                     |
| Image size        | 640                                    |
| Final model file  | `models/yolo11n_v6_best.pt`            |
| Summary file      | `results/yolo11n_v6_summary.json`      |
| Class result file | `results/yolo11n_v6_class_results.csv` |
| Model info file   | `results/yolo11n_v6_model_info.json`   |

## 5. 전체 성능 결과

test set 기준 전체 성능은 다음과 같다.

| Metric       | Score |
| ------------ | ----: |
| Precision    | 0.835 |
| Recall       | 0.719 |
| mAP@0.5      | 0.776 |
| mAP@0.5:0.95 | 0.545 |

전체적으로 Precision은 0.835로 비교적 안정적인 편이다. 반면 Recall은 0.719로, 일부 객체가 실제로 존재하더라도 모델이 탐지하지 못하는 경우가 발생할 수 있다.

본 프로젝트는 손끝으로 그림책 속 객체를 가리켰을 때 해당 객체를 탐지하고 설명을 제공하는 구조이므로, 실제 서비스 관점에서는 탐지 누락을 줄이는 것이 중요하다. 따라서 향후 모델 개선 시에는 Recall 개선과 bbox 정밀도 개선을 함께 고려할 필요가 있다.

## 6. 클래스별 성능 결과

test set 기준 클래스별 성능은 다음과 같다.

| Class             | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ----------------- | -----: | --------: | --------: | -----: | ------: | -----------: |
| all               |     96 |       129 |     0.835 |  0.719 |   0.776 |        0.545 |
| book_flower       |     11 |        11 |     1.000 |  0.937 |   0.995 |        0.539 |
| book_flowerpot    |     12 |        18 |     0.855 |  0.778 |   0.771 |        0.495 |
| book_monkey       |     14 |        21 |     0.718 |  0.729 |   0.746 |        0.488 |
| book_stone        |      9 |        14 |     0.836 |  0.365 |   0.678 |        0.342 |
| braille           |     21 |        22 |     0.934 |  0.864 |   0.913 |        0.758 |
| tactile_flower    |      4 |         4 |     0.597 |  0.500 |   0.495 |        0.420 |
| tactile_flowerpot |     10 |        11 |     0.704 |  0.727 |   0.808 |        0.644 |
| tactile_monkey    |      9 |        10 |     1.000 |  0.717 |   0.795 |        0.611 |
| tactile_stone     |      5 |         5 |     0.934 |  0.800 |   0.795 |        0.705 |
| text              |     12 |        13 |     0.770 |  0.775 |   0.759 |        0.450 |

## 7. 클래스별 성능 해석

### 7.1 상대적으로 안정적인 클래스

다음 클래스는 mAP@0.5 또는 mAP@0.5:0.95 기준으로 비교적 안정적인 성능을 보였다.

* `braille`
* `tactile_stone`
* `tactile_flowerpot`
* `tactile_monkey`
* `book_flower`

특히 `braille` 클래스는 mAP@0.5 0.913, mAP@0.5:0.95 0.758로 전체 클래스 중 가장 안정적인 탐지 성능을 보였다. 이는 점자 영역이 다른 객체에 비해 형태와 위치가 비교적 명확하게 구분되었기 때문으로 볼 수 있다.

### 7.2 개선이 필요한 클래스

다음 클래스는 Recall 또는 mAP@0.5:0.95 기준으로 개선이 필요하다.

* `book_stone`
* `tactile_flower`
* `text`
* `book_monkey`

`book_stone`은 Precision은 0.836으로 낮지 않지만 Recall이 0.365로 매우 낮다. 즉, 모델이 `book_stone`으로 예측한 경우에는 비교적 맞는 편이나, 실제 `book_stone` 객체를 놓치는 경우가 많을 가능성이 있다.

`tactile_flower`는 test instance 수가 4개로 매우 적고, Precision 0.597, Recall 0.500, mAP@0.5 0.495로 낮게 나타났다. 이는 데이터 수 부족, 촉각 교구 형태의 다양성 부족, 라벨링 기준 불안정 등의 영향을 받았을 수 있다.

`text` 클래스는 mAP@0.5:0.95가 0.450으로 상대적으로 낮다. 이는 텍스트 영역 자체는 탐지하더라도 bbox가 정확하게 맞지 않는 경우가 있음을 의미한다.

## 8. YOLOv5s baseline과 YOLO11n v6 비교

기존 YOLOv5s baseline 결과와 YOLO11n v6 결과는 다음과 같다.

| Model   | Dataset     | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 비고            |
| ------- | ----------- | --------: | -----: | ------: | -----------: | ------------- |
| YOLOv5s | Roboflow v2 |     0.847 |  0.889 |   0.898 |        0.641 | 초기 baseline   |
| YOLO11n | Roboflow v6 |     0.835 |  0.719 |   0.776 |        0.545 | 전체 통합 데이터셋 기준 |

수치상으로는 YOLOv5s baseline이 Recall과 mAP에서 더 높게 나타난다. 그러나 두 결과는 동일 데이터셋 기준이 아니므로 직접적인 성능 우열로 해석하기 어렵다.

YOLOv5s baseline은 Roboflow v2 기준으로 학습 및 평가되었고, YOLO11n은 Roboflow v6 통합 데이터셋 기준으로 학습 및 평가되었다. Roboflow v6는 클래스 수와 데이터 구성이 달라졌기 때문에, 난이도와 평가 조건이 v2와 동일하지 않다.

따라서 공정한 모델 비교를 위해서는 동일한 Roboflow v6 데이터셋에서 YOLOv5s, YOLO11n, YOLO11s 등의 모델을 같은 조건으로 재학습한 뒤 비교해야 한다.

## 9. 현재 모델의 장점과 한계

### 9.1 장점

* YOLO11n 모델을 사용하여 비교적 가벼운 객체 탐지 모델을 확보하였다.
* Roboflow v6 통합 데이터셋 기준으로 10개 클래스를 모두 포함하여 학습하였다.
* `braille`, `book_flower`, `tactile_stone` 등 일부 클래스에서는 안정적인 탐지 성능을 보였다.
* `best.pt`, summary JSON, class results CSV, model info JSON을 정리하여 후속 작업에서 활용할 수 있도록 했다.

### 9.2 한계

* Recall이 0.719로 탐지 누락 가능성이 존재한다.
* `book_stone`, `tactile_flower` 클래스의 성능이 낮다.
* `text` 클래스는 bbox 정밀도 개선이 필요하다.
* YOLOv5s baseline과의 비교는 데이터셋 버전 차이로 인해 직접 비교가 어렵다.
* 실제 웹캠 환경에서는 조명, 각도, 손가락 가림, 페이지 기울어짐 등의 영향으로 성능이 더 낮아질 수 있다.

## 10. 향후 개선 방향

현재 YOLO11n v6 모델은 기본적인 객체 탐지 기능 검증에는 사용할 수 있으나, 최종 적용 모델로 확정하기 전 추가 실험이 필요하다.

우선적으로 다음 작업을 검토할 수 있다.

1. 동일한 Roboflow v6 데이터셋 기준으로 YOLO11s 모델 추가 학습
2. YOLO11n 모델의 학습 설정 변경 실험
3. `book_stone`, `tactile_flower`, `text` 클래스 중심 데이터 보완
4. 클래스별 데이터 수 불균형 확인
5. confidence threshold별 탐지 결과 비교
6. 실제 웹캠 입력 환경에서 탐지 품질 확인

최종 모델 선정 시에는 단순히 mAP만 보는 것이 아니라, 다음 기준을 함께 고려해야 한다.

| 기준           | 이유                                  |
| ------------ | ----------------------------------- |
| Precision    | 잘못된 객체를 탐지하는 오탐을 줄이기 위함             |
| Recall       | 실제 객체를 놓치는 미탐을 줄이기 위함               |
| mAP@0.5      | 객체를 대략적으로 잘 잡는지 확인하기 위함             |
| mAP@0.5:0.95 | bbox가 정확하게 객체 영역을 감싸는지 확인하기 위함      |
| 모델 크기        | 백엔드 배포 및 로딩 부담을 줄이기 위함              |
| 추론 속도        | 웹캠 기반 실시간 서비스 적용 가능성을 판단하기 위함       |
| 약한 클래스 개선 여부 | 실제 서비스에서 특정 객체만 계속 누락되는 문제를 방지하기 위함 |

## 11. 최종 정리

YOLO11n v6 모델은 Roboflow v6 통합 데이터셋 기준으로 학습 및 평가되었으며, test set 기준 Precision 0.835, Recall 0.719, mAP@0.5 0.776, mAP@0.5:0.95 0.545의 성능을 보였다.

현재 모델은 기본적인 객체 탐지 기능 검증에는 사용할 수 있다. 다만 일부 클래스에서 탐지 누락과 bbox 정밀도 문제가 확인되었으므로, 최종 적용 전 추가 성능 개선 실험과 데이터 보완 검토가 필요하다.

특히 기존 YOLOv5s baseline과의 비교는 데이터셋 버전 차이로 인해 참고용으로만 해석해야 하며, 공정한 비교를 위해서는 동일한 Roboflow v6 데이터셋에서 후보 모델들을 재학습한 뒤 최종 모델을 선정하는 과정이 필요하다.

## 12. 관련 파일

* `models/yolo11n_v6_best.pt`
* `notebooks/yolo11n_v6_training.ipynb`
* `results/yolo11n_v6_summary.json`
* `results/yolo11n_v6_class_results.csv`
* `results/yolo11n_v6_model_info.json`

## 13. 관련 이슈

* Related to #24
* Related to #35