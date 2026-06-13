# Dataset Structure

## 개요

본 프로젝트는 AI 그림책 보조 서비스의 데이터셋 구조를 정의한다.

서비스는 사용자가 손가락으로 가리키는 동화책의 그림, 촉각 교구, 점자, 텍스트를 인식한 뒤 음성 안내를 제공하는 것을 목표로 한다. 페이지별로 서로 다른 설명을 제공하기 위해 페이지 인식 모델과 객체 인식 모델을 분리하여 관리한다.

- `page_classifier`: 현재 보고 있는 동화책 페이지를 분류
- `object_classifier`: 페이지 내 객체를 Bounding Box 단위로 탐지

---

## 전체 디렉터리 구조

```text
AI-Data/
├── README.md
├── docs/
│   └── dataset_structure.md
└── dataset/
    ├── page_classifier/
    │   ├── train/
    │   │   ├── page1/
    │   │   ├── page2/
    │   │   ├── page3/
    │   │   └── none/
    │   ├── valid/
    │   │   ├── page1/
    │   │   ├── page2/
    │   │   ├── page3/
    │   │   └── none/
    │   └── test/
    │       ├── page1/
    │       ├── page2/
    │       ├── page3/
    │       └── none/
    └── object_classifier/
        ├── images/
        │   ├── train/
        │   ├── valid/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── valid/
        │   └── test/
        └── data.yaml
```

---

## page_classifier

`page_classifier`는 사용자가 현재 보고 있는 동화책 페이지를 분류하기 위한 이미지 분류 데이터셋이다.

- Roboflow 프로젝트 유형: `Classification`
- 입력 데이터: 페이지 전체가 보이는 이미지
- 출력 클래스: `page1`, `page2`, `page3`, `none`

### 클래스 정의

| 클래스 | 정의 |
| --- | --- |
| `page1` | 동화책 1페이지가 주된 피사체로 촬영된 이미지 |
| `page2` | 동화책 2페이지가 주된 피사체로 촬영된 이미지 |
| `page3` | 동화책 3페이지가 주된 피사체로 촬영된 이미지 |
| `none` | page1, page2, page3 중 어떤 페이지로도 분류하기 어려운 이미지 |

### page_classifier none 기준

`none` 클래스에는 다음 데이터를 포함한다.

- 빈 배경
- 책 표지 또는 대상 페이지가 아닌 페이지
- 책 여백만 크게 보이는 이미지
- 손만 나온 이미지
- 대상 페이지를 식별하기 어려운 흐림, 흔들림, 과노출 이미지
- 촬영 실패 이미지

---

## object_classifier

`object_classifier`는 동화책 페이지 안의 객체를 탐지하기 위한 Object Detection 데이터셋이다.

- Roboflow 프로젝트 유형: `Object Detection`
- 입력 데이터: 객체가 포함된 페이지 이미지 또는 실제 사용 환경 이미지
- 출력 클래스: 책 속 그림, 촉각 교구, 점자, 텍스트 영역

### 클래스 정의

| 클래스 | 정의 |
| --- | --- |
| `book_monkey` | 책에 인쇄된 원숭이 그림 |
| `book_flower` | 책에 인쇄된 꽃 그림 |
| `book_flowerpot` | 책에 인쇄된 화분 그림 |
| `book_stone` | 책에 인쇄된 돌 그림 |
| `tactile_monkey` | 촉각 교구 형태의 원숭이 객체 |
| `tactile_flower` | 촉각 교구 형태의 꽃 객체 |
| `tactile_flowerpot` | 촉각 교구 형태의 화분 객체 |
| `tactile_stone` | 촉각 교구 형태의 돌 객체 |
| `braille` | 점자 영역 전체 |
| `text` | 일반 인쇄 텍스트 영역 |

### object_classifier none 사용 여부

Object Detection 모델에서는 일반적으로 `none` 클래스를 별도 라벨로 만들지 않는다. 객체가 없는 이미지는 라벨 파일을 비워 두거나 라벨 파일 없이 관리하여 negative sample로 활용한다.

따라서 `object_classifier`의 기본 클래스 목록에는 `none`을 포함하지 않는다.

단, Roboflow 프로젝트 운영 과정에서 negative sample을 명시적으로 분리해야 하는 경우에는 별도 태그 또는 데이터 관리 기준으로만 사용하고, 탐지 클래스에는 포함하지 않는다.

---

## 데이터 수집 기준

실제 사용 환경을 반영하기 위해 다음 조건을 포함하여 데이터를 수집한다.

- 다양한 조명: 밝은 실내, 어두운 실내, 그림자, 역광
- 다양한 거리: 가까운 거리, 중간 거리, 페이지 전체가 보이는 거리
- 다양한 각도: 정면, 기울어진 시점, 측면 시점
- 손 포함: 손가락이 객체를 가리키는 상황
- 손 미포함: 기준 이미지 확보용
- 부분 가림: 손, 촉각 교구, 페이지 접힘 등으로 객체 일부가 가려진 상황
- 실제 사용 환경: 책상 위, 무릎 위, 교구와 함께 놓인 상태
- 촬영 품질 다양성: 약한 흔들림, 약한 흐림 등 실제 입력에서 발생 가능한 상태

촬영 실패로 인해 페이지 또는 객체를 식별할 수 없는 이미지는 `page_classifier`의 `none` 또는 object detection negative sample로만 사용한다.

---

## 파일명 규칙

개인 이름, 촬영자 이름, 개인정보는 파일명에 포함하지 않는다.

### page_classifier 파일명

페이지 분류 데이터는 클래스명을 접두어로 사용하고 4자리 연속 번호를 붙인다.

```text
page1_0001.jpg
page1_0002.jpg
page2_0001.jpg
page3_0001.jpg
none_0001.jpg
```

저장 예시는 다음과 같다.

```text
dataset/page_classifier/train/page1/page1_0001.jpg
dataset/page_classifier/train/page1/page1_0002.jpg
dataset/page_classifier/valid/none/none_0001.jpg
```

### object_classifier 파일명

객체 인식 데이터는 이미지의 대표 객체명을 접두어로 사용하고 4자리 연속 번호를 붙인다. 한 이미지에 여러 객체가 있어도 파일명은 대표 객체 또는 촬영 목적 기준으로 정한다.

```text
book_monkey_0001.jpg
book_monkey_0002.jpg
tactile_monkey_0001.jpg
tactile_monkey_0002.jpg
braille_0001.jpg
text_0001.jpg
```

이미지와 라벨 파일은 같은 파일명을 사용하고 확장자만 다르게 관리한다.

```text
dataset/object_classifier/images/train/book_monkey_0001.jpg
dataset/object_classifier/labels/train/book_monkey_0001.txt
dataset/object_classifier/images/train/tactile_monkey_0001.jpg
dataset/object_classifier/labels/train/tactile_monkey_0001.txt
```

---

## Bounding Box 라벨링 기준

`object_classifier`는 YOLO 형식의 Bounding Box 라벨을 기준으로 한다.

- 객체 전체가 보이는 범위로 Bounding Box를 지정한다.
- 객체가 일부 가려진 경우에는 실제로 보이는 영역을 기준으로 Bounding Box를 지정한다.
- 하나의 이미지에 여러 객체가 존재할 경우 각각 독립적인 Bounding Box로 라벨링한다.
- 책 속 그림(`book_`)과 촉각 교구(`tactile_`)는 반드시 구분하여 라벨링한다.
- 점자는 점자 영역 전체를 하나의 `braille` 객체로 라벨링한다.
- 텍스트는 문장 또는 문단 단위로 `text` 객체를 라벨링한다.
- 손끝, 손가락, 손바닥은 객체 클래스로 라벨링하지 않는다.
- 손끝 좌표가 필요한 경우 모델 라벨이 아니라 MediaPipe Hands 등 별도 방식으로 추출한다.

### 라벨링 예시

`book_monkey_0001.jpg` 이미지에 다음 객체가 함께 포함되어 있는 경우:

- `book_monkey`
- `tactile_flowerpot`
- `braille`
- `text`

각 객체를 하나의 이미지 안에서 각각 독립적인 Bounding Box로 라벨링한다.

---

## Train / Validation / Test 분할 기준

데이터셋은 학습, 검증, 테스트 용도를 분리하여 관리한다.

권장 분할 비율은 다음과 같다.

| 구분 | 비율 | 용도 |
| --- | --- | --- |
| Train | 70% | 모델 학습 |
| Validation | 20% | 학습 중 성능 검증 및 하이퍼파라미터 조정 |
| Test | 10% | 최종 성능 평가 |

분할 시 다음 기준을 따른다.

- 같은 촬영 조건의 연속 이미지는 한 split에 몰리지 않도록 분산한다.
- 각 클래스가 train, valid, test에 최대한 고르게 포함되도록 한다.
- 손 포함 이미지와 손 미포함 이미지가 각 split에 모두 포함되도록 한다.
- 실제 사용 환경 이미지가 test set에 반드시 포함되도록 한다.
- 데이터 수가 적은 초기 단계에서는 test set을 과도하게 작게 만들지 않는다.

---

## data.yaml

`object_classifier/data.yaml`은 YOLO 학습 기준에 맞춰 클래스 목록을 정의한다.

```yaml
train: ../images/train
val: ../images/valid
test: ../images/test

nc: 10
names:
  - book_monkey
  - book_flower
  - book_flowerpot
  - book_stone
  - tactile_monkey
  - tactile_flower
  - tactile_flowerpot
  - tactile_stone
  - braille
  - text
```

---

## 클래스별 예시 이미지 정리 기준

클래스별 예시 이미지는 모델 학습 전 데이터 검수와 라벨링 기준 통일을 위해 관리한다.

권장 위치는 다음과 같다.

```text
dataset/examples/
├── page_classifier/
│   ├── page1/
│   ├── page2/
│   ├── page3/
│   └── none/
└── object_classifier/
    ├── book_monkey/
    ├── book_flower/
    ├── book_flowerpot/
    ├── book_stone/
    ├── tactile_monkey/
    ├── tactile_flower/
    ├── tactile_flowerpot/
    ├── tactile_stone/
    ├── braille/
    └── text/
```

각 클래스 폴더에는 대표 예시, 애매한 예시, 라벨링 제외 예시를 구분해 정리한다.

---

## 문서 관리

데이터셋 구축 과정에서 다음 문서를 함께 관리한다.

- `README.md`: 데이터셋 레포지토리 개요와 문서 링크
- `docs/dataset_structure.md`: 데이터셋 구조, 클래스, 수집 및 라벨링 기준
- 클래스 정의 문서: 클래스별 예시 이미지와 라벨링 판단 기준

클래스가 추가되거나 라벨링 기준이 변경되면 `data.yaml`과 문서를 함께 업데이트한다.
