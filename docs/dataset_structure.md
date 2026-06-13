# Dataset Structure

## 개요

본 프로젝트는 촉각형 그림책 기반 객체 인식 시스템을 개발하기 위한 데이터셋을 관리한다.

데이터셋은 페이지 인식과 객체 인식을 분리하여 구성한다.

---

## page_classifier

사용자가 현재 보고 있는 동화책 페이지를 분류하기 위한 데이터셋

### 클래스

- page1
- page2
- page3
- none

---

## object_classifier

동화책 내부 객체를 인식하기 위한 데이터셋

### 클래스

- book_monkey
- book_flowerpot
- book_stone
- book_flower
- tactile_monkey
- tactile_flowerpot
- tactile_stone
- tactile_flower
- braille
- text
- none

---

## none 클래스

다음 데이터를 포함한다.

- 빈 배경
- 책 여백
- 손만 나온 부분
- 대상이 아닌 그림
- 촬영 실패 이미지

---

## 데이터 수집 기준

- 다양한 조명
- 다양한 거리
- 다양한 각도
- 손 포함
- 손 미포함
- 실제 사용 환경 반영

---

## 파일명 규칙

개인 이름은 파일명에 포함하지 않는다.

모델 학습에서 범용적으로 사용할 수 있도록 이미지 파일명은 의미를 최소화한 고유 번호 형식을 사용한다.
라벨 정보는 클래스 디렉터리 또는 라벨 파일에서 관리한다.

### 이미지 분류

클래스별 디렉터리에 이미지를 저장하고, 파일명은 6자리 연속 번호를 사용한다.

예시

```text
dataset/page_classifier/train/page1/000001.jpg
dataset/page_classifier/train/page1/000002.jpg
dataset/page_classifier/train/none/000001.jpg
```

### 객체 인식

이미지와 라벨 파일은 같은 파일명을 사용하고 확장자만 다르게 관리한다.

예시

```text
dataset/object_classifier/images/train/000001.jpg
dataset/object_classifier/labels/train/000001.txt
dataset/object_classifier/images/train/000002.jpg
dataset/object_classifier/labels/train/000002.txt
```

---

## 라벨링 기준

- 객체 전체를 포함하도록 라벨링
- 점자는 점자 전체 영역 포함
- 텍스트는 문장 전체 영역 포함
- 손끝은 라벨링하지 않음
- 손끝 좌표는 MediaPipe Hands를 통해 추출
