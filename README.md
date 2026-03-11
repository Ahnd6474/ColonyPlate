# ColonyPlate

배양 접시(plate) 이미지에서 콜로니를 검출하고, 라벨을 반자동으로 재사용할 수 있는 도구입니다.  
YOLOv11 검출 결과를 기반으로 클릭 라벨링, 세션 저장/불러오기, 이미지 간 라벨 재매핑을 제공합니다.

## 이것이 무엇을 위한 프로젝트인가?
콜로니를 플레이트 뚜껑에 라벨링하는 방법은 번거롭고 문제가 많다.

- 콜로니 플레이트의 뚜껑이 돌아가는 경우
- 라벨이 실험실의 물질에 의해 지워지는 경우
- 콜로니의 경계가 명확하지 않은 경우

ColonyPlate는 이 병목을 줄이기 위해 만들어졌습니다. 핵심 목적은

1. 모델(YOLO)로 콜로니 후보를 먼저 빠르게 찾고,
2. 사용자가 최소한의 클릭으로 라벨을 보정/확정하고,
3. 다음 이미지에서는 이전 세션 라벨을 좌표 정합으로 자동 재매핑하여,
4. 최종 결과를 구조화된 형태(JSON/CSV)로 내보내는 것

입니다.

즉, 완전 자동 판독기가 아니라, **"검출 자동화 + 사람 검증"**에 최적화된 반자동 라벨링/추적 워크플로우 도구입니다.

## 주요 기능
- **YOLOv11 기반 콜로니 검출**
- **클릭/드롭다운 기반 라벨링**
- **세션 저장(`session_*.json`) 및 재사용**
- **이전 세션 라벨 자동 재매핑**
  - RANSAC + 유사변환(회전/이동/스케일)
  - Hungarian matching
- **라벨 라이브러리 기반 추천**
- **CSV 내보내기**

## 프로젝트 구조
```text
ColonyPlate/
├── colony_tool_yolo11_detect.py   # 실행 진입점 (CLI)
├── colony_tool/
│   ├── app.py                     # Gradio UI + 워크플로우
│   ├── detection.py               # ROI/검출/필터링
│   ├── matching.py                # 세션 간 좌표 정합/라벨 재매핑
│   ├── models.py                  # Dataclass(Det, Session)
│   ├── session_io.py              # 세션/라벨 라이브러리 I/O
│   └── utils.py                   # 공통 유틸
├── requirements.txt
└── yolov11/...
```

## 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행
```bash
python colony_tool_yolo11_detect.py \
  --weights yolov11/runs/detect/train/weights/best.pt \
  --sessions ./sessions \
  --host 127.0.0.1 \
  --port 7860
```

실행 후 브라우저에서 `http://127.0.0.1:7860` 접속.

## 사용 순서
1. 이미지 업로드 후 **이미지 로드 + 검출** 실행
2. 결과 이미지 클릭(또는 드롭다운 선택) 후 라벨 입력/추천 선택
3. **현재 세션 저장**으로 `session_*.json` 생성
4. 다음 이미지에서
   - 특정 세션 재매핑: **선택한 이전 세션으로 재매핑**
   - 자동 선택 재매핑: **best 세션 찾아 재매핑**
5. 필요 시 **CSV 내보내기**로 결과 추출

## 파이프라인 상세 설명 (행렬 기반)
ColonyPlate의 세션 간 재매핑은 "점 집합 정합(point set alignment)" 문제로 볼 수 있습니다.

### 1) 검출과 좌표 추출
- 입력 이미지에서 YOLO가 콜로니 박스를 예측합니다.
- 각 콜로니를 중심점 $p_i=(x_i,y_i)$로 표현해 점 집합 $P=p_i$를 만듭니다.

### 2) 이전 세션 좌표와 현재 좌표의 기하 정합
- 이전 세션 점 집합을 $Q=\{q_j\}$, 현재 이미지를 $P$라고 하면,
- 다음 유사변환(similarity transform)으로 $Q\to P$를 맞춥니다.


```math
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}
=
\underbrace{\begin{bmatrix}
s\cos\theta & -s\sin\theta & t_x \\
s\sin\theta &  s\cos\theta & t_y \\
0 & 0 & 1
\end{bmatrix}}_{T}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
```

- 여기서 $s$는 스케일, $\theta$는 회전, $t_x,t_y$는 평행이동입니다.
- RANSAC으로 이상치(outlier)를 제외하며 변환 행렬 $T$를 robust하게 추정합니다.

### 3) 거리 행렬(cost matrix) 생성
- 변환된 이전 점 $\hat{q}_j=Tq_j$와 현재 점 $p_i$ 사이 거리를 계산하여 비용행렬 vC$를 만듭니다.

```math
C_{ij} = \|p_i - \hat{q}_j\|_2
```

- 즉 $C$의 각 원소는 "현재 점 i와 이전 라벨 j를 매칭할 때의 비용"입니다.

### 4) Hungarian matching으로 1:1 최적 할당
- 비용행렬 $C$에 Hungarian 알고리즘을 적용해 총 비용이 최소가 되는 매칭을 찾습니다.
- 허용 거리(threshold)를 넘는 매칭은 버려 잘못된 라벨 전이를 줄입니다.

### 5) 라벨 재사용 + 사용자 보정
- 매칭 성공한 점은 이전 라벨을 자동으로 가져옵니다.
- 미매칭/의심 매칭은 사용자가 클릭으로 즉시 수정할 수 있습니다.

### 간단한 행렬 예시
현재 점이 3개, 이전 점이 3개라고 할 때 비용행렬이 아래처럼 나왔다고 가정하면:

```math
C =
\begin{bmatrix}
2.1 & 15.4 & 10.2 \\
13.8 & 1.9 & 12.5 \\
9.7 & 11.2 & 2.4
\end{bmatrix}
```

- Hungarian 결과는 일반적으로 대각선 성분(2.1, 1.9, 2.4)을 선택하는 할당이 됩니다.
- 이때 각 행(현재 검출)과 열(이전 라벨)이 1:1로 묶이며, 임계값 이내인 경우에만 라벨을 전이합니다.

## 라이선스
`LICENSE` 파일을 참고하세요.
