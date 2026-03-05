from pathlib import Path
from typing import Dict, List, Optional

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

from colony_tool.detection import detect_plate_roi, filter_dets, run_yolo_detect
from colony_tool.matching import remap_labels, score_session_for_current
from colony_tool.models import Det, Session
from colony_tool.session_io import (
    load_library,
    load_session,
    save_session,
    suggest_labels,
    update_library_from_session,
)
from colony_tool.utils import now_ms, to_gray, to_rgb


def draw_overlay(img_bgr: np.ndarray, dets: List[Det], labels: Dict[str, str], selected_id: Optional[str] = None) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, map(round, d.bbox_xyxy))
        color = (0, 255, 255) if d.det_id != selected_id else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cx, cy = map(int, map(round, d.centroid_xy))
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

        tag = labels.get(d.det_id, "")
        if tag:
            cv2.putText(out, tag, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def pick_det_by_click(dets: List[Det], x: float, y: float) -> Optional[Det]:
    inside = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside.append(d)
    if inside:
        return min(inside, key=lambda dd: (dd.centroid_xy[0] - x) ** 2 + (dd.centroid_xy[1] - y) ** 2)
    if not dets:
        return None
    return min(dets, key=lambda dd: (dd.centroid_xy[0] - x) ** 2 + (dd.centroid_xy[1] - y) ** 2)


def build_app(weights: str, sessions_dir: str):
    sessions_dir = Path(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    lib = load_library(sessions_dir)
    state = {
        "img_bgr": None,
        "img_name": "",
        "roi_circle": None,
        "dets": [],
        "labels": {},
        "selected_id": None,
        "last_session_path": None,
        "lib": lib,
    }

    def _render():
        if state["img_bgr"] is None:
            return None
        over = draw_overlay(state["img_bgr"], state["dets"], state["labels"], state["selected_id"])
        return to_rgb(over)

    def load_image(file, conf, iou, imgsz, min_area, max_area, use_roi, drop_ring):
        if file is None:
            return None, gr.update(value="(no image)"), gr.update(choices=[], value=None), gr.update(value=""), gr.update(value="")

        img_bgr = cv2.imread(file, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, gr.update(value="(read failed)"), gr.update(choices=[], value=None), gr.update(value=""), gr.update(value="")

        state["img_bgr"] = img_bgr
        state["img_name"] = Path(file).name
        roi = detect_plate_roi(to_gray(img_bgr)) if use_roi else None
        state["roi_circle"] = roi

        dets = run_yolo_detect(model, img_bgr, conf=conf, iou=iou, imgsz=int(imgsz))
        dets = filter_dets(
            dets,
            img_bgr.shape,
            roi_circle=roi,
            min_area_px=int(min_area),
            max_area_px=int(max_area),
            min_conf=float(conf),
            drop_edge_ring=bool(drop_ring),
        )
        state["dets"] = dets
        state["labels"] = {}
        state["selected_id"] = None

        choices = [f"{i:03d} | {d.det_id} | conf={d.conf:.2f}" for i, d in enumerate(dets)]
        return _render(), f"{state['img_name']} | dets={len(dets)}", gr.update(choices=choices, value=None), "", ""

    def select_by_dropdown(choice):
        if choice is None or state["img_bgr"] is None:
            state["selected_id"] = None
            return _render(), "", gr.update(choices=[])

        det_id = choice.split("|")[1].strip()
        state["selected_id"] = det_id

        det = next((d for d in state["dets"] if d.det_id == det_id), None)
        suggestions = []
        if det is not None and det.feat is not None:
            sug = suggest_labels(state["lib"], det.feat, topk=6)
            suggestions = [f"{name} (dist={dist:.3f})" for name, dist in sug]
        return _render(), det_id, gr.update(choices=suggestions, value=(suggestions[0] if suggestions else None))

    def on_click(evt: gr.SelectData):
        if state["img_bgr"] is None:
            return _render(), "", gr.update(choices=[])

        x, y = evt.index
        det = pick_det_by_click(state["dets"], float(x), float(y))
        if det is None:
            state["selected_id"] = None
            return _render(), "", gr.update(choices=[])

        state["selected_id"] = det.det_id
        suggestions = []
        if det.feat is not None:
            sug = suggest_labels(state["lib"], det.feat, topk=6)
            suggestions = [f"{name} (dist={dist:.3f})" for name, dist in sug]
        return _render(), det.det_id, gr.update(choices=suggestions, value=(suggestions[0] if suggestions else None))

    def apply_label(label_text, suggestion_choice):
        if state["img_bgr"] is None or state["selected_id"] is None:
            return _render(), "선택된 det 없음"

        label = (label_text or "").strip()
        if not label and suggestion_choice:
            label = suggestion_choice.split(" (dist=")[0].strip()

        if not label:
            return _render(), "라벨이 비어있음"

        state["labels"][state["selected_id"]] = label
        return _render(), f"라벨 적용: {state['selected_id']} -> {label}"

    def clear_labels():
        state["labels"] = {}
        return _render(), "라벨 초기화"

    def save_current_session():
        if state["img_bgr"] is None:
            return "이미지 없음"

        sess = Session(
            image_name=state["img_name"],
            image_size=(int(state["img_bgr"].shape[1]), int(state["img_bgr"].shape[0])),
            created_ms=now_ms(),
            dets=state["dets"],
            labels=state["labels"],
        )
        out = sessions_dir / f"session_{Path(state['img_name']).stem}_{sess.created_ms}.json"
        save_session(out, sess)
        state["last_session_path"] = str(out)

        update_library_from_session(sessions_dir, sess)
        state["lib"] = load_library(sessions_dir)

        return f"저장됨: {out.name} (labels={len(sess.labels)})"

    def load_and_remap(prev_json_path, ransac_iters, nn_thresh, max_match_dist, merge_mode):
        if state["img_bgr"] is None:
            return _render(), "(현재 이미지 없음)"
        if not prev_json_path:
            return _render(), "(이전 세션 json 선택 필요)"

        prev = load_session(Path(prev_json_path))
        remapped, _, score, missing, total = remap_labels(
            prev,
            state["dets"],
            ransac_iters=int(ransac_iters),
            nn_thresh=float(nn_thresh),
            max_match_dist=float(max_match_dist),
        )

        if merge_mode == "keep":
            for k, v in remapped.items():
                if k not in state["labels"]:
                    state["labels"][k] = v
        else:
            state["labels"].update(remapped)

        msg = f"재매핑: {len(remapped)}/{total} 성공, missing={missing}, ransac_res={score:.2f}"
        return _render(), msg

    def auto_recall_best(ransac_iters, nn_thresh, max_match_dist, merge_mode):
        if state["img_bgr"] is None:
            return _render(), "(현재 이미지 없음)"
        jsons = sorted(sessions_dir.glob("session_*.json"))
        if not jsons:
            return _render(), "(sessions 폴더에 session_*.json 없음)"

        best_key = None
        best_path = None
        for p in jsons:
            prev = load_session(p)
            s = score_session_for_current(
                prev,
                state["dets"],
                ransac_iters=int(ransac_iters),
                nn_thresh=float(nn_thresh),
                max_match_dist=float(max_match_dist),
            )
            if s is None:
                continue

            key = (s["found"], s["ratio"], -s["score"])
            if best_key is None or key > best_key:
                best_key = key
                best_path = p

        if best_path is None:
            return _render(), "(적절한 이전 세션을 못 찾음)"

        prev = load_session(best_path)
        remapped, _, score, missing, total = remap_labels(
            prev,
            state["dets"],
            ransac_iters=int(ransac_iters),
            nn_thresh=float(nn_thresh),
            max_match_dist=float(max_match_dist),
        )

        if merge_mode == "keep":
            for k, v in remapped.items():
                if k not in state["labels"]:
                    state["labels"][k] = v
        else:
            state["labels"].update(remapped)

        msg = f"[AUTO] best={best_path.name} | {len(remapped)}/{total} 성공, missing={missing}, ransac_res={score:.2f}"
        return _render(), msg

    def export_labels_csv():
        if state["img_bgr"] is None:
            return "(이미지 없음)"

        rows = ["det_id,label,cx,cy,x1,y1,x2,y2,conf,cls"]
        id2det = {d.det_id: d for d in state["dets"]}
        for det_id, label in state["labels"].items():
            d = id2det.get(det_id)
            if d is None:
                continue
            x1, y1, x2, y2 = d.bbox_xyxy
            cx, cy = d.centroid_xy
            rows.append(f"{det_id},{label},{cx:.2f},{cy:.2f},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{d.conf:.3f},{d.cls}")
        return "\n".join(rows)

    with gr.Blocks(title="Colony Tool (YOLOv11 Detect)") as demo:
        gr.Markdown(
            "## Colony 작업 툴 (YOLOv11 Detect 기반)\n"
            "- 검출 → 클릭 라벨링 → 세션 저장\n"
            "- 다음 사진에서 (회전/확대/이동) 라벨 재불러오기\n"
            "- 라벨 라이브러리로 자동 추천/불러오기"
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.File(label="이미지 파일 선택 (png/jpg)")
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="conf")
                iou = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="iou")
                imgsz = gr.Slider(320, 2048, value=1024, step=32, label="imgsz")
                min_area = gr.Slider(1, 10000, value=30, step=1, label="min bbox area (px^2)")
                max_area = gr.Slider(1000, 2_000_000, value=500_000, step=1000, label="max bbox area (px^2)")
                use_roi = gr.Checkbox(value=True, label="원형 ROI 마스킹(권장)")
                drop_ring = gr.Checkbox(value=True, label="가장자리 반사띠 주변 drop")
                btn_load = gr.Button("이미지 로드 + 검출")

                status = gr.Textbox(label="상태", value="")
                det_dropdown = gr.Dropdown(label="det 선택(드롭다운)", choices=[], value=None)
                selected_id = gr.Textbox(label="선택된 det_id", value="", interactive=False)

                suggestion = gr.Dropdown(label="라벨 추천(라이브러리)", choices=[], value=None)
                label_text = gr.Textbox(label="라벨 입력(빈칸이면 추천 사용)", value="")
                btn_apply = gr.Button("라벨 적용")
                btn_clear = gr.Button("라벨 초기화")

                btn_save = gr.Button("현재 세션 저장(session_*.json)")
                save_msg = gr.Textbox(label="저장 로그", value="", interactive=False)

                gr.Markdown("---")
                prev_json = gr.File(label="이전 세션 JSON 선택 (session_*.json)")
                ransac_iters = gr.Slider(50, 3000, value=400, step=50, label="RANSAC iters")
                nn_thresh = gr.Slider(3, 80, value=18, step=1, label="RANSAC NN thresh(px)")
                max_match_dist = gr.Slider(3, 120, value=25, step=1, label="매칭 허용 거리(px)")
                merge_mode = gr.Radio(["overwrite", "keep"], value="overwrite", label="병합 방식")

                btn_remap = gr.Button("선택한 이전 세션으로 재매핑")
                btn_auto = gr.Button("sessions 폴더에서 자동으로 best 세션 찾아 재매핑")
                remap_msg = gr.Textbox(label="재매핑 로그", value="", interactive=False)

                gr.Markdown("---")
                btn_export = gr.Button("라벨 CSV 내보내기(텍스트)")
                csv_out = gr.Textbox(label="CSV", value="", lines=10)

            with gr.Column(scale=2):
                img_out = gr.Image(label="결과(클릭해서 det 선택)", type="numpy")

        btn_load.click(
            fn=load_image,
            inputs=[img_in, conf, iou, imgsz, min_area, max_area, use_roi, drop_ring],
            outputs=[img_out, status, det_dropdown, selected_id, csv_out],
        )

        det_dropdown.change(fn=select_by_dropdown, inputs=[det_dropdown], outputs=[img_out, selected_id, suggestion])
        img_out.select(fn=on_click, inputs=None, outputs=[img_out, selected_id, suggestion])

        btn_apply.click(fn=apply_label, inputs=[label_text, suggestion], outputs=[img_out, status])
        btn_clear.click(fn=clear_labels, inputs=None, outputs=[img_out, status])

        btn_save.click(fn=save_current_session, inputs=None, outputs=[save_msg])

        btn_remap.click(
            fn=lambda f, a, b, c, m: load_and_remap(prev_json_path=f.name if f else "", ransac_iters=a, nn_thresh=b, max_match_dist=c, merge_mode=m),
            inputs=[prev_json, ransac_iters, nn_thresh, max_match_dist, merge_mode],
            outputs=[img_out, remap_msg],
        )

        btn_auto.click(
            fn=lambda a, b, c, m: auto_recall_best(a, b, c, m),
            inputs=[ransac_iters, nn_thresh, max_match_dist, merge_mode],
            outputs=[img_out, remap_msg],
        )

        btn_export.click(fn=export_labels_csv, inputs=None, outputs=[csv_out])

    return demo
