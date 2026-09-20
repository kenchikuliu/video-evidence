from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort


COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

BENCHMARK_OBJECT_CLASSES = (
    "person",
    "bus",
    "car",
    "truck",
    "sports ball",
    "laptop",
    "tv",
    "cell phone",
    "bottle",
    "cup",
    "chair",
    "dining table",
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo11n.onnx"


def sample_times(duration_seconds: float, interval_seconds: float) -> list[float]:
    if duration_seconds <= 0:
        return []
    if interval_seconds <= 0:
        raise ValueError("object sample interval must be positive")

    count = max(1, math.ceil(duration_seconds / interval_seconds))
    values = [
        round(index * interval_seconds, 3)
        for index in range(count)
        if index * interval_seconds < duration_seconds
    ]
    final = round(max(0.0, duration_seconds - 0.05), 3)
    if final - values[-1] >= min(0.25, interval_seconds / 2):
        values.append(final)
    return values


def create_session(model_path: Path) -> ort.InferenceSession:
    path = model_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YOLO ONNX model not found: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def prepare_image(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor[None]), scale, (pad_x, pad_y)


def postprocess(
    output: np.ndarray,
    original_shape: tuple[int, int],
    scale: float,
    padding: tuple[int, int],
    confidence_threshold: float,
    iou_threshold: float,
    allowed_classes: Sequence[str] | None = None,
) -> list[dict]:
    if output.ndim == 3 and output.shape[0] == 1:
        predictions = output[0]
    elif output.ndim == 2:
        predictions = output
    else:
        raise ValueError(f"unexpected YOLO output shape: {output.shape}")
    expected_columns = 4 + len(COCO_CLASSES)
    if predictions.shape[0] == expected_columns:
        predictions = predictions.T
    if predictions.shape[1] != expected_columns:
        raise ValueError(f"unexpected YOLO class count: {predictions.shape}")

    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    allowed_ids = None
    if allowed_classes:
        allowed = set(allowed_classes)
        allowed_ids = {index for index, label in enumerate(COCO_CLASSES) if label in allowed}

    candidates = []
    nms_boxes = []
    nms_scores = []
    original_height, original_width = original_shape
    pad_x, pad_y = padding
    for row, class_id, score in zip(predictions, class_ids, scores):
        class_id = int(class_id)
        score = float(score)
        if score < confidence_threshold or (allowed_ids is not None and class_id not in allowed_ids):
            continue
        center_x, center_y, width, height = (float(value) for value in row[:4])
        left = (center_x - width / 2 - pad_x) / scale
        top = (center_y - height / 2 - pad_y) / scale
        right = (center_x + width / 2 - pad_x) / scale
        bottom = (center_y + height / 2 - pad_y) / scale
        left = max(0.0, min(left, original_width - 1.0))
        top = max(0.0, min(top, original_height - 1.0))
        right = max(left, min(right, float(original_width)))
        bottom = max(top, min(bottom, float(original_height)))
        box = [left, top, right, bottom]
        candidates.append((class_id, score, box))
        nms_boxes.append([left, top, right - left, bottom - top])
        nms_scores.append(score)

    if not candidates:
        return []
    indexes = cv2.dnn.NMSBoxes(
        nms_boxes,
        nms_scores,
        confidence_threshold,
        iou_threshold,
    )
    kept = np.asarray(indexes).reshape(-1).tolist() if len(indexes) else []
    detections = []
    for index in kept:
        class_id, score, box = candidates[int(index)]
        detections.append(
            {
                "class_id": class_id,
                "label": COCO_CLASSES[class_id],
                "confidence": round(score, 5),
                "box": [round(value, 1) for value in box],
            }
        )
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def detect_frame(
    frame: np.ndarray,
    session: ort.InferenceSession,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.45,
    allowed_classes: Sequence[str] | None = None,
) -> list[dict]:
    model_input = session.get_inputs()[0]
    size = int(model_input.shape[-1])
    tensor, scale, padding = prepare_image(frame, size)
    output = session.run(None, {model_input.name: tensor})[0]
    return postprocess(
        output,
        frame.shape[:2],
        scale,
        padding,
        confidence_threshold,
        iou_threshold,
        allowed_classes,
    )


def detect_video_objects(
    video_path: Path,
    duration_seconds: float,
    model_path: Path = DEFAULT_MODEL_PATH,
    interval_seconds: float = 1.0,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.45,
    allowed_classes: Sequence[str] | None = None,
    session: ort.InferenceSession | None = None,
) -> dict:
    detector = session or create_session(model_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video for object detection: {video_path}")

    detections = []
    try:
        for timestamp in sample_times(duration_seconds, interval_seconds):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            items = detect_frame(
                frame,
                detector,
                confidence_threshold,
                iou_threshold,
                allowed_classes,
            )
            detections.append(
                {
                    "timestamp": timestamp,
                    "objects": items,
                    "count": len(items),
                }
            )
    finally:
        capture.release()

    unique_classes = sorted(
        {item["label"] for frame in detections for item in frame["objects"]}
    )
    return {
        "model": model_path.name,
        "runtime": "onnxruntime-cpu",
        "sample_interval_seconds": interval_seconds,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "classes_requested": list(allowed_classes or COCO_CLASSES),
        "detections": detections,
        "detection_count": sum(frame["count"] for frame in detections),
        "unique_classes": unique_classes,
    }
