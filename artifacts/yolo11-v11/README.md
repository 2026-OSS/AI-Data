# YOLO11n v11 Object Detection Artifact

Roboflow `ai-picture-book-object-detection` dataset version 11로 학습한 YOLO11n 객체 탐지 산출물입니다.

## Files

- `best.pt`: YOLO11n v11 best checkpoint
- `data.yaml`: class names and Roboflow dataset metadata

## Classes

`data.yaml` 기준 클래스 순서는 다음과 같습니다.

1. `book_flower`
2. `book_flowerpot`
3. `book_monkey`
4. `book_stone`
5. `braille`
6. `tactile_flower`
7. `tactile_flowerpot`
8. `tactile_monkey`
9. `tactile_stone`
10. `text`

## Webcam Test

```bash
cd AI-Data
bash scripts/run_yolo11_webcam.sh
```

손끝으로 가리킨 객체를 선택하는 테스트는 다음처럼 실행합니다.

```bash
cd AI-Data
HAND_MODE=point bash scripts/run_yolo11_webcam.sh
```
