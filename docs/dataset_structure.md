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

예시

page1_yein_001.jpg

book_monkey_narin_001.jpg

braille_yeowon_001.jpg

---

## 라벨링 기준

- 객체 전체를 포함하도록 라벨링
- 점자는 점자 전체 영역 포함
- 텍스트는 문장 전체 영역 포함
- 손끝은 라벨링하지 않음
- 손끝 좌표는 MediaPipe Hands를 통해 추출
