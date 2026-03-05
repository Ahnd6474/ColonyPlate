# ColonyPlate

배양 접시(plate) 이미지에서 콜로니를 검출하고, 라벨을 반자동으로 재사용할 수 있는 도구입니다.  
YOLOv11 검출 결과를 기반으로 클릭 라벨링, 세션 저장/불러오기, 이미지 간 라벨 재매핑을 제공합니다.

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

## 디버깅 팁
- 검출이 과다/부족하면 `conf`, `iou`, `imgsz` 조정
- 가장자리 오검출이 많으면 `원형 ROI`, `drop_ring` 활성화 유지
- 재매핑 성능이 낮으면 `RANSAC iters`, `NN thresh`, `매칭 허용 거리`를 이미지 품질에 맞게 조절
- 레이블 관련 디버깅 필요
## requirements 점검 메모
- `streamlit`은 현재 코드에서 사용하지 않아 제거했습니다.
- `ultralytics` 실행 환경 안정화를 위해 `torch`를 명시했습니다.
- 나머지 패키지는 실제 import 경로 기준으로 정리했습니다.

## 라이선스
`LICENSE` 파일을 참고하세요.
