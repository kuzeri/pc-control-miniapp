import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

PERSON_CLASS = 0
BAG_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}


def center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def is_person_nearby(bag_bbox, people_bboxes, proximity_px):
    cx, cy = center(bag_bbox)
    for pbbox in people_bboxes:
        px, py = center(pbbox)
        if np.sqrt((cx - px) ** 2 + (cy - py) ** 2) < proximity_px:
            return True
    return False


def process_video(video_path, threshold_seconds, proximity_px):
    if video_path is None:
        return None, "Загрузите видео для анализа."

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    threshold_frames = int(threshold_seconds * fps)

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    unattended_counters = {}
    alerted_ids = set()
    events = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model.track(frame, persist=True, verbose=False, classes=list(BAG_CLASSES.keys()) + [PERSON_CLASS])

        if results[0].boxes is None or results[0].boxes.id is None:
            out.write(frame)
            continue

        boxes = results[0].boxes
        people_bboxes = []
        bag_detections = []

        for box in boxes:
            cls = int(box.cls[0])
            bbox = box.xyxy[0].cpu().numpy()
            track_id = int(box.id[0])

            if cls == PERSON_CLASS:
                people_bboxes.append(bbox)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
            elif cls in BAG_CLASSES:
                bag_detections.append({
                    "bbox": bbox,
                    "class": BAG_CLASSES[cls],
                    "track_id": track_id,
                })

        current_ids = set()
        for bag in bag_detections:
            tid = bag["track_id"]
            current_ids.add(tid)
            bbox = bag["bbox"]

            nearby = is_person_nearby(bbox, people_bboxes, proximity_px)
            if nearby:
                unattended_counters[tid] = 0
            else:
                unattended_counters[tid] = unattended_counters.get(tid, 0) + 1

            frames_unattended = unattended_counters.get(tid, 0)
            secs_unattended = frames_unattended / fps

            x1, y1, x2, y2 = map(int, bbox)

            if frames_unattended >= threshold_frames:
                color = (0, 0, 255)
                label = f"ABANDONED {bag['class']} ({secs_unattended:.0f}s)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                if tid not in alerted_ids:
                    alerted_ids.add(tid)
                    ts = f"{frame_idx / fps:.1f}s"
                    events.append(f"[{ts}] ТРЕВОГА: {bag['class']} без присмотра {threshold_seconds}+ сек")
            elif not nearby:
                color = (0, 165, 255)
                label = f"{bag['class']} ({secs_unattended:.0f}s)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                color = (0, 200, 0)
                label = bag["class"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for tid in list(unattended_counters.keys()):
            if tid not in current_ids:
                del unattended_counters[tid]

        out.write(frame)

    cap.release()
    out.release()

    if not events:
        log = "Оставленных предметов не обнаружено."
    else:
        log = "\n".join(events)

    return out_path, log


with gr.Blocks(title="Abandoned Object Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# Детектор оставленных предметов
Загрузи видео — система найдёт сумки, рюкзаки и чемоданы, оставленные без присмотра.

**Цветовая схема:**
- 🟢 Зелёный — предмет рядом с хозяином
- 🟠 Оранжевый — хозяин отошёл, идёт отсчёт
- 🔴 Красный — ТРЕВОГА, предмет без присмотра
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Загрузить видео")
            threshold = gr.Slider(
                minimum=3, maximum=60, value=10, step=1,
                label="Порог тревоги (секунды без присмотра)"
            )
            proximity = gr.Slider(
                minimum=50, maximum=400, value=150, step=10,
                label="Радиус близости (пиксели)"
            )
            btn = gr.Button("Анализировать", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Результат")
            log_out = gr.Textbox(
                label="Журнал событий",
                lines=8,
                placeholder="Здесь появятся события тревоги..."
            )

    btn.click(
        fn=process_video,
        inputs=[video_in, threshold, proximity],
        outputs=[video_out, log_out],
    )

    gr.Examples(
        examples=[],
        inputs=video_in,
    )

demo.launch()
