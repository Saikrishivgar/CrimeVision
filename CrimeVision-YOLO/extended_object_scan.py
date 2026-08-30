"""On-demand YOLOE scanning that writes into CrimeVision's shared track shape."""

import time
import uuid

import cv2
import numpy as np

from config import (
    OPEN_VOCAB_CONFIDENCE_THRESHOLD,
    OPEN_VOCAB_FRAME_STRIDE,
    OPEN_VOCAB_MAX_GAP_SECONDS,
    OPEN_VOCAB_MIN_BBOX_IOU,
    OPEN_VOCAB_MIN_HITS,
    OPEN_VOCAB_MIN_TRACK_DURATION,
    OPEN_VOCAB_STATIONARY_SPEED,
)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1.0)


def _dedupe_frame_detections(detections):
    """Collapse overlapping synonym detections before temporal aggregation."""
    result = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if any(_iou(detection["bbox"], existing["bbox"]) >= 0.50 for existing in result):
            continue
        result.append(detection)
    return result


def _track_stability(points):
    if len(points) < 2:
        return 0.0
    values = [
        _iou(points[index - 1]["bbox"], points[index]["bbox"])
        for index in range(1, len(points))
    ]
    return sum(values) / len(values) if values else 0.0


def _describe_object(record, model_registry):
    attributes = {
        "object_color": "Unknown",
        "vehicle_color": "Unknown",
        "shirt_color": "Unknown",
        "vehicle_make": "Unknown",
        "vehicle_model": "Unknown",
    }
    crops = record.get("crops_buffer", [])
    if crops and model_registry and model_registry.get("attribute_extractor"):
        extractor = model_registry["attribute_extractor"]
        description = extractor.describe_crop(crops[0][1])
        color = extractor.extract_color_from_text(description, "Unknown")
        attributes["object_color"] = color

    color = attributes["object_color"]
    label = record["object"]
    description = f"{color} {label}" if color != "Unknown" else label
    return attributes, description.capitalize()


def _make_embedding(record, description, model_registry):
    embedder = model_registry.get("embedder") if model_registry else None
    if not embedder:
        return None

    text_embedding = embedder.embed_text(description)
    crops = record.get("crops_buffer", [])
    image_embedding = embedder.embed_crop(crops[0][2]) if crops else None
    if image_embedding is not None and text_embedding is not None:
        combined = image_embedding + text_embedding
        norm = np.linalg.norm(combined)
        return (combined / norm).tolist() if norm else None
    if image_embedding is not None:
        return image_embedding.tolist()
    return text_embedding.tolist() if text_embedding is not None else None


def run_extended_scan(
    source,
    video_id,
    query,
    detector,
    model_registry=None,
    conf=None,
    frame_stride=None,
):
    """Run a query-scoped YOLOE scan without rerunning YOLO11."""
    processor = None
    started = time.time()
    conf = OPEN_VOCAB_CONFIDENCE_THRESHOLD if conf is None else conf
    frame_stride = OPEN_VOCAB_FRAME_STRIDE if frame_stride is None else frame_stride
    object_name = query.get("object") or "object"
    concepts = query.get("concepts") or [object_name]
    time_start = query.get("time_start")
    time_end = query.get("time_end")
    histories = {}
    synthetic_id = 200000

    try:
        from video_processor import VideoProcessor

        processor = VideoProcessor(source)
        for frame_idx, timestamp, frame in processor.read_frames():
            current_seconds = frame_idx / processor.fps
            if frame_idx % frame_stride != 0:
                continue
            if time_start is not None and current_seconds < time_start:
                continue
            if time_end is not None and current_seconds > time_end:
                continue

            detections = detector.detect_and_track(
                frame,
                concepts=concepts,
                persist=True,
                conf=conf,
            )
            detections = _dedupe_frame_detections(
                [d for d in detections if d["confidence"] >= conf]
            )

            for detection in detections:
                raw_track_id = detection.get("track_id")
                if raw_track_id is None:
                    synthetic_id += 1
                    raw_track_id = synthetic_id

                key = (object_name, raw_track_id)
                history = histories.get(key)
                if history is None:
                    history = {
                        "track_id": raw_track_id,
                        "object": object_name,
                        "first_seen": current_seconds,
                        "last_seen": current_seconds,
                        "confidences": [],
                        "centers": [],
                        "points": [],
                        "crops_buffer": [],
                        "last_detection_time": current_seconds,
                    }
                    histories[key] = history

                if current_seconds - history["last_detection_time"] > OPEN_VOCAB_MAX_GAP_SECONDS:
                    # A long gap means this cannot safely be one persistent object.
                    continue

                bbox = tuple(int(value) for value in detection["bbox"])
                confidence = float(detection["confidence"])
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                speed = 0.0
                if history["centers"]:
                    prev_x, prev_y, prev_time = history["centers"][-1]
                    dt = current_seconds - prev_time
                    if dt > 0:
                        speed = ((cx - prev_x) ** 2 + (cy - prev_y) ** 2) ** 0.5 / dt

                history["confidences"].append(confidence)
                history["centers"].append((cx, cy, current_seconds))
                history["last_seen"] = current_seconds
                history["last_detection_time"] = current_seconds
                history["points"].append(
                    {
                        "timestamp": current_seconds,
                        "frame": frame_idx,
                        "bbox": list(bbox),
                        "confidence": confidence,
                        "speed": round(speed, 2),
                        "source_model": "yoloe",
                    }
                )
                if crop.size:
                    history["crops_buffer"].append((confidence, crop.copy(), crop.copy()))
                    history["crops_buffer"] = sorted(
                        history["crops_buffer"], key=lambda item: item[0], reverse=True
                    )[:5]

        records = []
        for history in histories.values():
            points = history["points"]
            duration = history["last_seen"] - history["first_seen"]
            average_confidence = sum(history["confidences"]) / len(history["confidences"])
            stability = _track_stability(points)
            average_speed = (
                sum(point["speed"] for point in points) / len(points) if points else 0.0
            )
            max_speed = max((point["speed"] for point in points), default=0.0)

            # Temporal consistency is mandatory for a persistent arbitrary-object track.
            if len(points) < OPEN_VOCAB_MIN_HITS:
                continue
            if duration < OPEN_VOCAB_MIN_TRACK_DURATION and len(points) < OPEN_VOCAB_MIN_HITS + 1:
                continue
            if len(points) > 1 and stability < OPEN_VOCAB_MIN_BBOX_IOU:
                continue

            attributes, description = _describe_object(history, model_registry)
            stationary = average_speed <= OPEN_VOCAB_STATIONARY_SPEED
            record = {
                "event_id": f"evt_{uuid.uuid4().hex[:8]}",
                "track_id": history["track_id"],
                "track_uid": None,
                "object": object_name.capitalize(),
                "timestamp": history["first_seen"],
                "frame": points[0]["frame"],
                "confidence": round(average_confidence, 4),
                "bbox": points[-1]["bbox"],
                "clip_start": max(0.0, history["first_seen"] - 3.0),
                "clip_end": min(processor.duration, history["last_seen"] + 3.0),
                "speed": None if stationary else round(average_speed, 2),
                "speed_unit": "Stationary" if stationary else "pixels/second",
                "stationary": stationary,
                "action": "Unknown",
                "action_confidence": 0.0,
                "first_seen": history["first_seen"],
                "last_seen": history["last_seen"],
                "points": points,
                "average_speed": round(average_speed, 2),
                "max_speed": round(max_speed, 2),
                "brand": "Unknown",
                "model": "Unknown",
                "brand_confidence": 0.0,
                "model_confidence": 0.0,
                "source_model": "yoloe",
                "query_concepts": concepts,
                "attributes": attributes,
                "color": attributes.get("object_color", "Unknown"),
                "description": description,
            }
            record["embedding"] = _make_embedding(record, description, model_registry)
            records.append(record)

        return {
            "video_duration": processor.duration,
            "detections": records,
            "source_model": "yoloe",
            "query": query,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    finally:
        if processor is not None:
            processor.release()
        detector.reset_tracking()
