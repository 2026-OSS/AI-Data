# Dataset Preprocessing, Augmentation, and Versioning

## 개요

본 문서는 YOLO 모델 학습에 사용할 데이터셋의 전처리, 증강, 데이터 분할 검수, Roboflow Version 생성 기준을 정의한다.

대상 이슈는 `#4 [DATA] 데이터 전처리·증강 및 데이터셋 버전 생성`이다.

본 프로젝트는 실제 서비스 환경에서 사용자가 동화책과 촉각 교구를 다양한 조명, 거리, 각도에서 촬영하고 손가락으로 객체를 가리키는 상황을 전제로 한다. 따라서 모델이 특정 촬영 조건에 과적합되지 않도록 Roboflow Generate 기능을 활용하여 증강 데이터를 생성하고, 학습에 사용할 데이터셋 Version을 명확히 관리한다.

---

## 작업 범위

### 데이터 전처리

학습에 부적합한 이미지를 제거하고 라벨 오류를 수정한다.

- 이미지 품질 검수
- 중복 이미지 제거
- 잘못된 라벨 수정
- Bounding Box 위치 검수
- 클래스 누락 여부 확인

### 데이터 증강

Roboflow Generate 기능을 활용하여 실제 촬영 환경에서 발생할 수 있는 변화를 반영한다.

- `90° Rotation`
- `Rotation`
- `Brightness`
- `Blur`
- `Noise`

### 데이터셋 분할

학습, 검증, 테스트 데이터가 목적에 맞게 분리되어 있는지 확인한다.

| Split | 권장 비율 | 용도 |
| --- | ---: | --- |
| Train | 70% | 모델 학습 |
| Validation | 20% | 학습 중 성능 검증 |
| Test | 10% | 최종 성능 평가 |

### 데이터셋 Version 생성

전처리, 증강, 분할 검수가 완료된 데이터셋을 Roboflow Version으로 생성하고 YOLO 형식으로 Export한다.

- Roboflow Version 생성
- `data.yaml` 확인
- YOLO Format Export 확인
- 증강 결과 검수

---

## 데이터 전처리 기준

### 이미지 품질 검수

다음 이미지는 학습 데이터에서 제외하거나 별도 검토 대상으로 분리한다.

- 객체 또는 페이지를 식별할 수 없을 정도로 심하게 흐린 이미지
- 과도한 흔들림이 있는 이미지
- 객체가 대부분 가려진 이미지
- 과노출 또는 저노출로 객체 경계가 보이지 않는 이미지
- 개인정보, 얼굴, 이름표 등 불필요한 식별 정보가 포함된 이미지
- 촬영 대상과 무관한 배경 또는 물체가 주된 피사체인 이미지

단, 실제 서비스 환경에서 발생할 수 있는 약한 흐림, 약한 그림자, 약한 기울어짐은 데이터 다양성을 위해 포함할 수 있다.

### 중복 이미지 제거

동일하거나 거의 동일한 이미지는 데이터 편중을 만들 수 있으므로 제거한다.

- 같은 장면을 연속 촬영한 이미지가 특정 split에 과도하게 몰리지 않도록 한다.
- 초점, 각도, 객체 위치가 거의 같은 이미지는 대표 이미지만 남긴다.
- 손가락 위치만 아주 미세하게 다른 연속 이미지는 필요한 수량만 유지한다.
- 중복 제거 후 클래스별 이미지 수가 과도하게 줄어들지 않았는지 확인한다.

### 잘못된 라벨 수정

라벨 오류는 학습 성능에 직접 영향을 주므로 Version 생성 전에 수정한다.

- 이미지 안의 모든 대상 객체가 라벨링되었는지 확인한다.
- `book_` 클래스와 `tactile_` 클래스가 서로 바뀌지 않았는지 확인한다.
- `braille`과 `text`가 각각 점자 영역과 일반 인쇄 텍스트 영역에 지정되었는지 확인한다.
- Bounding Box가 객체 외곽에 맞게 지정되었는지 확인한다.
- 가려진 객체는 실제로 보이는 영역 기준으로 Bounding Box가 지정되었는지 확인한다.

---

## Roboflow Generate 증강 기준

증강은 실제 서비스 환경에서 발생할 수 있는 변화만 반영한다. 객체 형태를 과도하게 왜곡하거나 라벨 의미를 바꿀 수 있는 증강은 적용하지 않는다.

| 증강 | 목적 | 적용 기준 |
| --- | --- | --- |
| `90° Rotation` | 카메라 방향 변화 대응 | 페이지 방향이 바뀐 입력 가능성을 반영 |
| `Rotation` | 촬영 각도와 기울어짐 대응 | 약한 회전 위주로 적용 |
| `Brightness` | 조명 변화 대응 | 밝은 실내, 어두운 실내, 그림자 상황 반영 |
| `Blur` | 약한 흔들림과 초점 흐림 대응 | 객체 식별이 가능한 수준만 허용 |
| `Noise` | 카메라 센서 노이즈와 저조도 대응 | 라벨 경계가 무너지지 않는 수준만 허용 |

### 증강 적용 주의사항

- 증강 후에도 Bounding Box가 객체를 정확히 감싸는지 확인한다.
- 점자와 텍스트가 읽기 어려울 정도로 blur 또는 noise가 강하면 제외한다.
- 과도한 rotation으로 페이지가 화면 밖으로 잘리는 경우 제외한다.
- 너무 어둡거나 밝아 객체 경계가 사라진 이미지는 제외한다.
- 증강 데이터가 원본보다 과도하게 많아져 특정 클래스나 촬영 조건을 지배하지 않도록 한다.

---

## 데이터셋 분할 검수 기준

### Split 비율 확인

Roboflow에서 데이터셋을 생성하기 전에 split 비율을 확인한다.

| 항목 | 기준 |
| --- | --- |
| Train | 전체 데이터의 약 70% |
| Validation | 전체 데이터의 약 20% |
| Test | 전체 데이터의 약 10% |
| 클래스 비율 | 각 split에 주요 클래스가 고르게 포함되어야 함 |
| 손 포함 여부 | 손가락 포함/미포함 이미지가 각 split에 모두 포함되어야 함 |

데이터 수가 적은 초기 단계에서는 비율을 기계적으로 맞추기보다, test set에 실제 사용 환경 이미지가 충분히 들어가는지를 함께 확인한다.

### 클래스 비율 확인

다음 항목을 기준으로 클래스 편중 여부를 확인한다.

- 특정 클래스가 train에만 있고 validation 또는 test에 없는지 확인한다.
- `book_` 클래스와 `tactile_` 클래스의 수량 차이가 과도하지 않은지 확인한다.
- `braille`, `text`처럼 넓은 영역 클래스가 지나치게 적지 않은지 확인한다.
- page별로 등장하는 객체가 특정 page 이미지에만 과도하게 몰리지 않았는지 확인한다.

### 데이터 누수 확인

데이터 누수는 모델 평가 결과를 실제보다 좋게 보이게 만들 수 있으므로 반드시 검수한다.

- 같은 장면의 연속 촬영 이미지가 train, validation, test에 동시에 들어가지 않도록 한다.
- 원본 이미지와 해당 증강 이미지가 서로 다른 split에 나뉘지 않도록 한다.
- 거의 동일한 촬영 조건의 이미지가 test set에 과도하게 중복되지 않도록 한다.
- 동일 파일에서 파생된 이미지가 여러 split에 섞였는지 확인한다.

---

## Roboflow Version 생성 기준

### Version 생성 전 확인

Roboflow Version을 생성하기 전에 다음 항목을 확인한다.

- 전처리 대상 이미지 정리 완료
- 중복 이미지 제거 완료
- 라벨 누락 및 오라벨링 수정 완료
- Bounding Box 위치 검수 완료
- Train / Validation / Test 분할 확인 완료
- 적용할 증강 옵션 확정

### Version 생성

Roboflow `Generate` 단계에서 전처리와 증강 설정을 적용한 뒤 Version을 생성한다.

권장 Version 메모에는 다음 정보를 포함한다.

```text
object_classifier vN
- Split: train 70 / valid 20 / test 10
- Augmentation: 90deg rotation, rotation, brightness, blur, noise
- Classes: 10
- Notes: hand included / real service camera conditions included
```

### 현재 생성된 Roboflow Version 기록

Roboflow에서 생성한 데이터셋 Version은 학습 재현성을 위해 문서에 기록한다. Version마다 이미지 수, split 비율, 전처리, 증강 설정이 다를 수 있으므로 학습에 사용한 Version을 명확히 남긴다.

| Roboflow Version | Dataset Version | 생성일 | 생성자 | 총 이미지 | Train | Validation | Test | 전처리 | 증강 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `v1` | `V1.0.1` | 2026-06-13 | 조예인 | 90 | 63장, 70% | 18장, 20% | 9장, 10% | Auto-Orient, Resize 640x640 | 없음 |
| `v2` | `V1.0.2` | 2026-06-13 | 조예인 | 720 | 630장, 88% | 60장, 8% | 30장, 4% | Auto-Orient, Resize 640x640 | Outputs per training example 3, Rotation, Grayscale, Brightness, Exposure, Blur, Noise |
| `v4` | `V1.0.4` | 2026-06-15 | 조예인 | 1656 | 1449장, 88% | 138장, 8% | 69장, 4% | Auto-Orient, Resize 640x640 | Outputs per training example 3, Rotation, Grayscale, Brightness, Exposure, Blur, Noise |

세부 설정은 다음과 같다.

| Version | 세부 설정 |
| --- | --- |
| `v1` | `Auto-Orient: Applied`, `Resize: Stretch to 640x640`, 증강 없음 |
| `v2` | `Auto-Orient: Applied`, `Resize: Stretch to 640x640`, `Rotation: -15° to +15°`, `Grayscale: 10%`, `Brightness: -20% to +20%`, `Exposure: -15% to +15%`, `Blur: up to 1px`, `Noise: up to 1.05% of pixels` |
| `v4` | `Auto-Orient: Applied`, `Resize: Stretch to 640x640`, `Rotation: -15° to +15°`, `Grayscale: 10%`, `Brightness: -20% to +20%`, `Exposure: -15% to +15%`, `Blur: up to 1px`, `Noise: up to 1.05% of pixels` |

`v1`은 권장 split인 Train 70%, Validation 20%, Test 10%와 일치하지만 증강이 적용되지 않았다. `v2`와 `v4`는 증강이 적용되어 이미지 수가 늘었으나 split 비율이 Train 88%, Validation 8%, Test 4%로 기록되어 있으므로, 최종 학습 Version으로 사용할 경우 평가 데이터가 충분한지 별도 검수한다.

### YOLO Export 확인

Version 생성 후 YOLO 형식으로 Export한다.

확인할 파일은 다음과 같다.

```text
train/images/
train/labels/
valid/images/
valid/labels/
test/images/
test/labels/
data.yaml
```

`data.yaml`의 클래스 수와 클래스 순서는 프로젝트 기준과 일치해야 한다.

```yaml
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

## 증강 결과 검수

Version 생성 후 샘플 이미지를 직접 확인한다.

| 검수 항목 | 기준 |
| --- | --- |
| 이미지 품질 | 객체와 페이지를 식별할 수 있어야 함 |
| Bounding Box | 증강 후에도 객체 외곽과 맞아야 함 |
| 클래스 라벨 | 증강 과정에서 라벨이 바뀌지 않아야 함 |
| 텍스트/점자 | blur, noise 후에도 영역 식별이 가능해야 함 |
| 회전 결과 | 객체가 과도하게 잘리지 않아야 함 |
| 클래스 분포 | 특정 클래스만 과도하게 증강되지 않아야 함 |

검수 중 문제가 발견되면 Version을 학습에 사용하지 않고, 원본 데이터 또는 Generate 설정을 수정한 뒤 새 Version을 생성한다.

---

## 체크리스트

### 데이터 전처리

- [ ] 이미지 품질 검수
- [ ] 중복 이미지 제거
- [ ] 잘못된 라벨 수정

### 데이터 증강

- [ ] `90° Rotation` 적용
- [ ] `Rotation` 적용
- [ ] `Brightness` 적용
- [ ] `Blur` 적용
- [ ] `Noise` 적용

### 데이터셋 분할

- [ ] Train 70% 확인
- [ ] Validation 20% 확인
- [ ] Test 10% 확인
- [ ] 클래스 비율 확인
- [ ] 데이터 누수 여부 확인

### 데이터셋 Version 생성

- [ ] Roboflow Version 생성
- [ ] `data.yaml` 확인
- [ ] YOLO Export 확인
- [ ] 증강 결과 검수

---

## 관련 문서

- [01_dataset_structure.md](01_dataset_structure.md): 데이터셋 구조 및 클래스 정의
- [02_dataset_collection.md](02_dataset_collection.md): 객체 탐지 데이터셋 수집 기준
- [03_dataset_labeling.md](03_dataset_labeling.md): Roboflow 데이터 라벨링 및 페이지별 음성 설명 매핑
