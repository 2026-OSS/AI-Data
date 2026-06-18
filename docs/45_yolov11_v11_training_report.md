# YOLO11n v11 학습 결과 리포트

## 1. 작업 개요

본 문서는 그림책 객체 탐지를 위한 YOLO11n v11 모델의 학습 및 평가 결과를 정리한 리포트이다.

YOLO11n v11 모델은 Roboflow v11 통합 데이터셋을 기준으로 학습되었으며, 학습 완료 모델은 `AI-Data/artifacts/yolo11-v11/best.pt`로 정리하였다. 본 문서는 v11 모델의 학습 조건, 전체 성능, 클래스별 성능 차이, 이전 YOLO11n v10 및 YOLOv5s baseline과의 비교 시 주의사항, 향후 개선 방향을 정리하는 것을 목적으로 한다.

## 2. 사용 데이터셋

| 항목              | 내용                                 |
| --------------- | ---------------------------------- |
| Dataset source  | Roboflow                           |
| Workspace       | `2026-oss`                         |
| Project         | `ai-picture-book-object-detection` |
| Dataset version | `v11`                              |
| Export format   | YOLOv8 format                      |
| Task type       | Object Detection                   |
| Image size      | 640                                |

Roboflow v11 데이터셋은 그림책 객체 탐지를 위한 최신 통합 데이터셋 기준으로 사용되었다. v11은 기존 v6, v10 데이터셋 이후 갱신된 버전이므로, 이전 모델 결과와 비교할 때는 데이터셋 구성과 test set 분포가 달라졌을 수 있음을 고려해야 한다.

따라서 v11 모델의 성능은 동일한 v11 데이터셋에서 재학습한 모델끼리 비교하는 것이 가장 공정하다. v6, v10, YOLOv5s baseline과의 비교는 실험 진행 과정에서의 참고용으로 해석한다.

## 3. 탐지 클래스

YOLO11n v11 모델은 총 10개 클래스를 기준으로 학습되었다.

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

| 항목                | 내용                                                |
| ----------------- | ------------------------------------------------- |
| Model             | YOLO11n                                           |
| Initial weight    | `yolo11n.pt`                                      |
| Dataset           | Roboflow v11                                      |
| Run name          | `picture_book_yolo11n_v11`                        |
| Epochs            | 100                                               |
| Batch size        | 16                                                |
| Image size        | 640                                               |
| Confidence        | 0.25                                              |
| Final model file  | `artifacts/yolo11-v11/best.pt`                    |
| Dataset yaml      | `artifacts/yolo11-v11/data.yaml`                  |
| Notebook          | `notebooks/yolo11n_v11_training.ipynb`            |
| Colab best path   | `/content/runs/detect/runs/detect/picture_book_yolo11n_v11/weights/best.pt` |

학습은 Ultralytics YOLO11n 모델을 기반으로 진행되었다. 노트북 실행 환경에서 확인된 모델 정보는 fused 기준 101 layers, 2,584,102 parameters, 6.3 GFLOPs이다.

## 5. 전체 성능 결과

test set 기준 전체 성능은 다음과 같다.

| Metric       | Score |
| ------------ | ----: |
| Precision    | 0.903 |
| Recall       | 0.857 |
| mAP@0.5      | 0.919 |
| mAP@0.5:0.95 | 0.708 |

v11 모델은 전체적으로 Precision 0.903, Recall 0.857을 기록하여 오탐과 미탐 양쪽 모두에서 비교적 안정적인 성능을 보였다. mAP@0.5는 0.919로 객체를 대략적으로 찾는 능력이 높게 나타났고, mAP@0.5:0.95도 0.708로 이전 실험 대비 bbox 정밀도가 개선된 편이다.

본 프로젝트는 손끝으로 그림책 속 객체를 가리켰을 때 해당 객체를 탐지하고 설명을 제공하는 구조이므로, 실제 서비스 관점에서는 Recall이 특히 중요하다. v11의 Recall 0.857은 v6, v10 대비 개선된 결과지만, 웹캠 환경에서는 손가락 가림, 조명, 각도, 페이지 기울어짐 등의 변수가 추가되므로 실제 입력 환경 검증이 필요하다.

## 6. 클래스별 성능 결과

test set 기준 클래스별 성능은 다음과 같다.

| Class             | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ----------------- | -----: | --------: | --------: | -----: | ------: | -----------: |
| all               |    101 |       109 |     0.903 |  0.857 |   0.919 |        0.708 |
| book_flower       |      5 |         5 |     0.931 |  1.000 |   0.995 |        0.561 |
| book_flowerpot    |     13 |        13 |     0.927 |  0.976 |   0.974 |        0.686 |
| book_monkey       |     11 |        11 |     1.000 |  0.813 |   0.968 |        0.752 |
| book_stone        |      5 |         5 |     1.000 |  0.866 |   0.995 |        0.860 |
| braille           |     15 |        15 |     0.790 |  0.867 |   0.863 |        0.715 |
| tactile_flower    |      7 |         7 |     0.783 |  0.857 |   0.855 |        0.648 |
| tactile_flowerpot |     10 |        12 |     0.818 |  0.751 |   0.815 |        0.689 |
| tactile_monkey    |     10 |        11 |     0.910 |  0.636 |   0.843 |        0.626 |
| tactile_stone     |      5 |         5 |     0.936 |  1.000 |   0.995 |        0.845 |
| text              |     25 |        25 |     0.933 |  0.800 |   0.882 |        0.695 |

## 7. 클래스별 성능 해석

### 7.1 상대적으로 안정적인 클래스

다음 클래스는 mAP@0.5 또는 mAP@0.5:0.95 기준으로 비교적 안정적인 성능을 보였다.

* `book_stone`
* `tactile_stone`
* `book_monkey`
* `braille`
* `text`
* `book_flowerpot`

특히 `book_stone`은 v6에서 Recall이 낮았던 클래스였으나, v11 test set에서는 Precision 1.000, Recall 0.866, mAP@0.5:0.95 0.860으로 크게 개선된 결과를 보였다. `tactile_stone` 역시 Recall 1.000, mAP@0.5:0.95 0.845로 안정적인 성능을 보였다.

`book_monkey`는 Precision 1.000, mAP@0.5:0.95 0.752로 bbox 정밀도까지 비교적 좋은 편이다. 다만 Recall은 0.813이므로 일부 누락 가능성은 남아 있다.

### 7.2 개선이 필요한 클래스

다음 클래스는 Recall 또는 mAP@0.5:0.95 기준으로 추가 확인이 필요하다.

* `tactile_monkey`
* `tactile_flower`
* `tactile_flowerpot`
* `book_flower`

`tactile_monkey`는 Precision 0.910으로 예측 자체는 비교적 정확하지만 Recall이 0.636으로 낮다. 즉, 모델이 `tactile_monkey`로 탐지한 경우에는 맞는 편이나 실제 객체를 놓치는 경우가 상대적으로 많을 수 있다.

`tactile_flower`와 `tactile_flowerpot`은 mAP@0.5는 각각 0.855, 0.815로 나쁘지 않지만, mAP@0.5:0.95가 0.648, 0.689 수준이다. 실제 손끝 지시 서비스에서는 bbox 위치가 손끝과의 거리 계산에 영향을 줄 수 있으므로, 촉각 객체류의 bbox 정밀도는 추가 검증이 필요하다.

`book_flower`는 mAP@0.5가 0.995이고 Recall도 1.000이지만, mAP@0.5:0.95는 0.561로 낮게 나타났다. 이는 객체 존재 여부는 잘 찾지만 bbox가 정밀하게 맞지 않는 사례가 있을 수 있음을 의미한다. 다만 test instance가 5개로 적으므로 데이터 수에 따른 변동성도 함께 고려해야 한다.

## 8. 이전 모델과의 비교

기존 YOLOv5s baseline, YOLO11n v6, YOLO11n v10, YOLO11n v11 결과는 다음과 같다.

| Model   | Dataset      | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 비고            |
| ------- | ------------ | --------: | -----: | ------: | -----------: | ------------- |
| YOLOv5s | Roboflow v2  |     0.847 |  0.889 |   0.898 |        0.641 | 초기 baseline   |
| YOLO11n | Roboflow v6  |     0.835 |  0.719 |   0.776 |        0.545 | v6 통합 데이터셋 기준 |
| YOLO11n | Roboflow v10 |     0.894 |  0.781 |   0.831 |        0.647 | v10 통합 데이터셋 기준 |
| YOLO11n | Roboflow v11 |     0.903 |  0.857 |   0.919 |        0.708 | v11 통합 데이터셋 기준 |

수치상으로 v11은 YOLO11n v6, v10 대비 모든 주요 지표에서 개선되었다. 특히 Recall은 v10의 0.781에서 v11의 0.857로 상승했고, mAP@0.5:0.95도 0.647에서 0.708로 개선되었다.

다만 위 비교는 데이터셋 버전이 서로 다르므로 모델 자체의 절대적인 우열로 해석하기에는 제한이 있다. Roboflow v11 기준으로 최종 모델을 선정하려면 YOLO11n, YOLO11s, YOLOv5s 등 후보 모델을 모두 v11 데이터셋에서 동일 조건으로 재학습한 뒤 비교하는 것이 바람직하다.

## 9. 현재 모델의 장점과 한계

### 9.1 장점

* YOLO11n 기반의 가벼운 객체 탐지 모델을 v11 데이터셋 기준으로 확보하였다.
* test set 기준 Precision 0.903, Recall 0.857로 이전 YOLO11n 실험 대비 성능이 개선되었다.
* mAP@0.5 0.919, mAP@0.5:0.95 0.708로 객체 탐지와 bbox 정밀도 양쪽에서 안정적인 결과를 보였다.
* `book_stone`, `tactile_stone`, `book_monkey` 등 주요 클래스에서 높은 mAP@0.5:0.95를 기록했다.
* 모델 산출물과 클래스 메타데이터를 `artifacts/yolo11-v11` 아래에 정리하여 백엔드 및 웹캠 테스트에서 활용할 수 있도록 했다.

### 9.2 한계

* `tactile_monkey`의 Recall이 0.636으로 낮아 실제 객체 누락 가능성이 있다.
* `book_flower`는 test instance가 5개로 적고 mAP@0.5:0.95가 0.561이므로 bbox 정밀도와 데이터 수를 추가 확인해야 한다.
* 일부 test class의 instance 수가 적어 클래스별 성능 수치의 변동성이 있을 수 있다.
* Roboflow v11 데이터셋에 segment 정보가 일부 섞여 있다는 경고가 있었으며, 평가 시 box 기준으로만 사용되었다.
* 실제 웹캠 환경에서는 조명, 각도, 손가락 가림, 페이지 기울어짐 등의 영향으로 notebook test set 성능보다 낮아질 수 있다.

## 10. 백엔드 및 웹캠 연동 관점

v11 모델의 입력과 출력 형태는 다음과 같이 정리할 수 있다.

| 항목   | 내용                                  |
| ---- | ----------------------------------- |
| Input | image 또는 webcam frame              |
| Image size | 640                           |
| Output class | detected object class name   |
| Output confidence | detection confidence score |
| Output bbox | `[x1, y1, x2, y2]`             |

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

손끝 지시 방식에서는 단순 탐지 성능뿐 아니라 bbox 중심점, 손끝 좌표와 bbox 간 거리, confidence threshold가 실제 사용성에 큰 영향을 준다. 따라서 v11 모델 적용 후에는 test set 지표와 별도로 웹캠 기반 정성 평가를 진행해야 한다.

## 11. 향후 개선 방향

현재 YOLO11n v11 모델은 이전 YOLO11n 실험 대비 성능이 개선되어 실제 기능 검증용 후보 모델로 사용할 수 있다. 다만 최종 적용 모델로 확정하기 전 다음 작업을 검토할 필요가 있다.

1. 실제 웹캠 입력 환경에서 손끝 지시 탐지 품질 확인
2. `tactile_monkey`, `tactile_flower`, `tactile_flowerpot` 중심 오탐 및 미탐 사례 확인
3. `book_flower` bbox 정밀도 개선 여부 확인
4. confidence threshold별 탐지 결과 비교
5. 동일 Roboflow v11 데이터셋 기준으로 YOLO11s 모델 추가 학습
6. YOLO11n v11 결과 파일을 `results/` 아래 JSON, CSV 형태로 함께 정리
7. test set 외 실제 촬영 이미지로 별도 검증 세트 구성

최종 모델 선정 시에는 단순히 mAP만 보는 것이 아니라 다음 기준을 함께 고려해야 한다.

| 기준           | 이유                                  |
| ------------ | ----------------------------------- |
| Precision    | 잘못된 객체를 탐지하는 오탐을 줄이기 위함             |
| Recall       | 실제 객체를 놓치는 미탐을 줄이기 위함               |
| mAP@0.5      | 객체를 대략적으로 잘 잡는지 확인하기 위함             |
| mAP@0.5:0.95 | bbox가 정확하게 객체 영역을 감싸는지 확인하기 위함      |
| 모델 크기        | 백엔드 배포 및 로딩 부담을 줄이기 위함              |
| 추론 속도        | 웹캠 기반 실시간 서비스 적용 가능성을 판단하기 위함       |
| 손끝 선택 정확도    | 탐지 bbox와 손끝 좌표가 실제 서비스 결과에 직접 연결되기 때문 |
| 약한 클래스 개선 여부 | 실제 서비스에서 특정 객체만 계속 누락되는 문제를 방지하기 위함 |

## 12. 최종 정리

YOLO11n v11 모델은 Roboflow v11 통합 데이터셋 기준으로 학습 및 평가되었으며, test set 기준 Precision 0.903, Recall 0.857, mAP@0.5 0.919, mAP@0.5:0.95 0.708의 성능을 보였다.

현재 결과만 보면 v11 모델은 이전 YOLO11n v6, v10 실험 대비 전반적으로 개선된 후보 모델이다. 특히 Recall과 mAP@0.5:0.95가 함께 상승하여 실제 탐지 누락과 bbox 정밀도 측면에서 더 안정적인 방향으로 개선되었다.

다만 `tactile_monkey`의 Recall, `book_flower`의 bbox 정밀도, 실제 웹캠 환경에서의 손끝 지시 성능은 추가 검증이 필요하다. 최종 적용 전에는 웹캠 테스트와 confidence threshold 조정, 약한 클래스 중심 데이터 보완 여부를 함께 확인하는 것이 좋다.

## 13. 관련 파일

* `artifacts/yolo11-v11/best.pt`
* `artifacts/yolo11-v11/data.yaml`
* `artifacts/yolo11-v11/README.md`
* `notebooks/yolo11n_v11_training.ipynb`
* `scripts/run_yolo11_webcam.sh`

## 14. 관련 이슈

* Related to #45
