# Dataset Labeling

## 개요

본 문서는 Roboflow를 활용한 데이터 라벨링 기준을 정의한다.

대상 이슈는 `#3 [DOCS] 데이터 라벨링`이다.

본 프로젝트는 현재 보고 있는 동화책 페이지에 따라 같은 객체라도 서로 다른 음성 설명을 제공해야 한다. 따라서 페이지 분류와 객체 탐지를 하나의 모델로 처리하지 않고, 다음 두 작업으로 분리하여 라벨링한다.

- `page_classifier`: 현재 보고 있는 동화책 페이지를 이미지 단위로 분류
- `object_classifier`: 페이지 내 그림, 촉각 교구, 점자, 텍스트를 Bounding Box 단위로 탐지

현재는 `object_classifier` 라벨링을 우선 진행한다. 이후 `page_classifier`용 Roboflow Classification 프로젝트를 생성하여 페이지 단위 라벨링을 추가로 진행한다.

---

## 라벨링 작업 범위

### object_classifier

`object_classifier`는 Roboflow `Object Detection` 프로젝트로 관리한다.

- 한 이미지 안에 여러 객체가 존재할 수 있다.
- 각 객체는 개별 Bounding Box로 라벨링한다.
- 손가락, 손바닥, 팔은 입력 상황을 구성하는 요소로만 사용하고 객체 라벨로 만들지 않는다.
- 객체가 손이나 교구에 일부 가려진 경우에는 실제로 보이는 영역을 기준으로 Bounding Box를 지정한다.

### page_classifier

`page_classifier`는 Roboflow `Classification` 프로젝트로 관리한다.

- 이미지 전체에 `page1`, `page2`, `page3`, `none` 중 하나의 라벨을 부여한다.
- 페이지 전체 또는 페이지를 식별할 수 있는 충분한 영역이 보이는 이미지를 사용한다.
- 대상 페이지를 식별하기 어려운 이미지, 손만 나온 이미지, 빈 배경 이미지는 `none`으로 분류한다.

---

## object_classifier 클래스

| 클래스 | 라벨링 대상 |
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

### Bounding Box 기준

- 객체 외곽을 최대한 정확히 감싸도록 지정한다.
- 배경 여백은 최소화한다.
- 객체가 일부 가려져 있으면 가려진 전체 형태를 추정하지 않고 보이는 부분만 감싼다.
- 책 속 그림 객체(`book_`)와 촉각 교구 객체(`tactile_`)는 반드시 구분한다.
- 점자는 줄 단위가 아니라 점자 영역 전체를 하나의 `braille`로 라벨링한다.
- 일반 인쇄 문장은 문장 또는 문단 단위로 `text` 라벨을 지정한다.

---

## page_classifier 클래스

| 클래스 | 라벨링 대상 |
| --- | --- |
| `page1` | 1페이지가 주된 피사체로 보이는 이미지 |
| `page2` | 2페이지가 주된 피사체로 보이는 이미지 |
| `page3` | 3페이지가 주된 피사체로 보이는 이미지 |
| `none` | 대상 페이지로 분류하기 어려운 이미지 |

### none 기준

다음 이미지는 `none`으로 라벨링한다.

- 빈 배경
- 손만 나온 이미지
- 대상 페이지가 아닌 책 표지 또는 다른 페이지
- 페이지 일부만 보여 `page1`, `page2`, `page3`를 구분하기 어려운 이미지
- 심한 흐림, 흔들림, 과노출로 페이지 식별이 어려운 이미지

---

## 페이지별 객체 음성 설명 매핑

페이지 분류 결과와 객체 탐지 결과를 함께 사용하여 최종 음성 설명을 선택한다.

```text
최종 음성 설명 = response_map[page_classifier 결과][object_classifier 결과]
```

예를 들어 현재 페이지가 `page1`이고 손가락이 가리킨 객체가 `book_monkey`로 인식되면, `page1.book_monkey`에 해당하는 문구를 출력한다.

### page1

| 객체 라벨 | 음성 설명 |
| --- | --- |
| `book_monkey` | 꼬마 원숭이입니다. |
| `book_flower` | 시들어가는 하얀 꽃 한 송이를 발견했어요. |
| `book_stone` | 돌이에요. 이 돌 뒤에 꼬마 원숭이가 있어요. |
| `tactile_monkey` | 부들부들한 원숭이예요. 손끝으로 천천히 만져보세요. |
| `tactile_flower` | 시들어가는 꽃이에요. 꽃잎의 모양을 손끝으로 만져보세요. |
| `tactile_stone` | 동글동글한 돌멩이예요. 손끝으로 천천히 만져보세요. |
| `braille` | 점자에요. 시들어가는 하얀 꽃 한 송이를 발견했어요. 길을 걷던 사람에게 밟혀 다친 것 같아요. |
| `text` | 시들어가는 하얀 꽃 한 송이를 발견했어요. 길을 걷던 사람에게 밟혀 다친 것 같아요. |

### page2

| 객체 라벨 | 음성 설명 |
| --- | --- |
| `book_monkey` | 꼬마 원숭이가 코코넛 화분에 꽃을 심고 있어요. |
| `book_flowerpot` | 코코넛으로 만든 화분이에요. 그 안에 흙을 담고 꽃을 심었어요. |
| `tactile_monkey` | 부들부들한 원숭이예요. 손끝으로 천천히 만져보세요. |
| `tactile_flowerpot` | 오돌토돌한 코코넛 화분이에요. 손끝으로 울퉁불퉁한 부분을 천천히 만져보세요. |
| `braille` | 점자 라벨이에요. 이야기 설명 |
| `text` | 꼬마 원숭이는 코코넛을 주워 반으로 쪼갠 다음, 그 안에 흙을 넣고 꽃을 심었어요. |

### page3

| 객체 라벨 | 음성 설명 |
| --- | --- |
| `book_monkey` | 꼬마 원숭이가 싱싱해진 꽃이 담긴 화분을 바라보고 있어요. |
| `book_flowerpot` | 꽃이 담긴 화분이에요. 꼬마 원숭이가 정성껏 돌본 꽃이에요. |
| `tactile_monkey` | 부들부들한 원숭이예요. 손끝으로 원숭이의 모양을 천천히 느껴보세요. |
| `tactile_flowerpot` | 오돌토돌한 코코넛 화분이에요. 손끝으로 울퉁불퉁한 부분을 천천히 만져보세요. |
| `braille` | 점자 라벨이에요. 이야기 전체. |
| `text` | 여러 날이 지나갔습니다. 꽃은 싱싱하게 살아났어요. |

---

## 라벨링 검수 기준

### 클래스 누락 검수

- 이미지 안의 모든 대상 객체가 라벨링되었는지 확인한다.
- 한 이미지에 여러 객체가 있으면 모든 객체에 개별 Bounding Box가 있어야 한다.
- `book_` 객체와 `tactile_` 객체가 서로 바뀌지 않았는지 확인한다.

### Bounding Box 위치 검수

- Bounding Box가 객체 외곽에 맞게 지정되었는지 확인한다.
- 배경 여백이 과도하게 포함되지 않았는지 확인한다.
- 객체 일부가 가려진 경우 보이는 영역 기준으로 라벨링되었는지 확인한다.
- 점자와 텍스트 영역이 서로 겹치거나 바뀌지 않았는지 확인한다.

### 오라벨링 검수

- 페이지별로 등장하지 않는 객체 라벨이 잘못 지정되지 않았는지 확인한다.
- `book_flower`와 `book_flowerpot`, `tactile_flower`와 `tactile_flowerpot`처럼 형태가 유사한 클래스가 혼동되지 않았는지 확인한다.
- `braille`과 `text`가 각각 점자 영역과 일반 인쇄 텍스트 영역에 맞게 적용되었는지 확인한다.

### 데이터 품질 검수

- 심하게 흔들리거나 흐린 이미지는 학습 데이터에서 제외한다.
- 개인정보, 얼굴, 불필요한 주변 물체가 포함된 이미지는 제외하거나 비식별 처리한다.
- 특정 조명, 각도, 배경에 데이터가 과도하게 몰리지 않도록 확인한다.
- 손가락 포함 이미지와 손가락 미포함 이미지가 모두 포함되도록 확인한다.

---

## 체크리스트

### object_classifier 라벨링

- [ ] `book_monkey` 라벨링
- [ ] `book_flower` 라벨링
- [ ] `book_flowerpot` 라벨링
- [ ] `book_stone` 라벨링
- [ ] `tactile_monkey` 라벨링
- [ ] `tactile_flower` 라벨링
- [ ] `tactile_flowerpot` 라벨링
- [ ] `tactile_stone` 라벨링
- [ ] `braille` 라벨링
- [ ] `text` 라벨링

### page_classifier 라벨링

- [ ] Roboflow Classification 프로젝트 생성
- [ ] `page1` 라벨링
- [ ] `page2` 라벨링
- [ ] `page3` 라벨링
- [ ] `none` 라벨링

### 검수

- [ ] 클래스 누락 검수
- [ ] Bounding Box 위치 검수
- [ ] 오라벨링 검수
- [ ] 데이터 품질 검수

---

## 관련 문서

- [01_dataset_structure.md](01_dataset_structure.md): 데이터셋 구조 및 클래스 정의
- [02_dataset_collection.md](02_dataset_collection.md): 객체 탐지 데이터셋 수집 기준
