# YOLOv5s v6 vs YOLO11n v6 성능 비교

## 1. 작업 목적

본 문서는 Roboflow v6 데이터셋을 기준으로 학습한 YOLOv5s 모델과 YOLO11n 모델의 성능을 비교하기 위해 작성하였다.

비교 기준은 test set 기준 전체 성능 지표이며, Precision, Recall, mAP@0.5, mAP@0.5:0.95를 사용하였다.

## 2. 비교 대상

| Model | Dataset | Model file |
|---|---|---|
| YOLOv5s v6 | Roboflow v6 | `artifacts/yolov5-v6/best.pt` |
| YOLO11n v6 | Roboflow v6 | `models/yolo11n_v6_best.pt` |

## 3. 전체 성능 비교

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLOv5s v6 | 0.917 | 0.847 | 0.885 | 0.660 |
| YOLO11n v6 | 0.835 | 0.719 | 0.776 | 0.545 |

## 4. 비교 결과

YOLOv5s v6는 Precision, Recall, mAP@0.5, mAP@0.5:0.95 모든 지표에서 YOLO11n v6보다 높은 성능을 보였다.

특히 Recall은 YOLOv5s v6가 0.847, YOLO11n v6가 0.719로 나타났다. 본 프로젝트는 손끝으로 가리킨 객체를 탐지한 뒤 설명과 TTS 출력으로 연결하는 구조이므로, 객체를 놓치지 않는 Recall 지표가 중요하다.

mAP@0.5:0.95 역시 YOLOv5s v6가 0.660, YOLO11n v6가 0.545로 나타났다. 이는 YOLOv5s v6가 전체적인 bounding box 정밀도 측면에서도 더 안정적인 결과를 보였음을 의미한다.

## 5. 결론

Roboflow v6 test set 기준 비교에서는 YOLOv5s v6가 YOLO11n v6보다 전반적으로 우수한 성능을 보였다.

따라서 v6 데이터셋 기준 후보 모델 비교에서는 YOLOv5s v6를 더 우선적인 후보로 볼 수 있다. 다만 최종 적용 모델은 실제 웹캠 입력 환경에서의 추론 속도, 탐지 안정성, 백엔드 연동 가능성을 함께 확인한 뒤 결정하는 것이 적절하다.
