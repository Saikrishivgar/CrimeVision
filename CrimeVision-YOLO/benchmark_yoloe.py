"""Small, annotation-aware YOLOE benchmark for the local CCTV videos.

Without annotations the script reports detections, runtime, and tracking
stability only. Precision/recall are intentionally reported as unavailable
instead of being inferred from model confidence.

Annotation format:
{
  "video.mp4": {
    "speaker": [[x1, y1, x2, y2], ...],
    "computer": [[x1, y1, x2, y2], ...]
  }
}
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2

YOLO_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(YOLO_DIR, "../CrimeVision-backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config import YOLOE_MODEL_PATH, OPEN_VOCAB_CONFIDENCE_THRESHOLD
from detectors.yoloe_detector import YOLOEDetector
from video_processor import VideoProcessor
from object_query import normalize_object_request


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1.0)


def score_boxes(predictions, ground_truth, threshold=0.5):
    unmatched = list(ground_truth)
    true_positive = 0
    for prediction in predictions:
        match = next((box for box in unmatched if iou(prediction, box) >= threshold), None)
        if match is not None:
            true_positive += 1
            unmatched.remove(match)
    false_positive = len(predictions) - true_positive
    false_negative = len(unmatched)
    return true_positive, false_positive, false_negative


def benchmark_video(video_path, detector, concepts, stride, conf):
    processor = VideoProcessor(video_path)
    detections = []
    track_hits = defaultdict(int)
    inference_seconds = 0.0
    sampled = 0
    try:
        for frame_idx, _, frame in processor.read_frames():
            if frame_idx % stride != 0:
                continue
            sampled += 1
            started = time.perf_counter()
            frame_detections = detector.detect_and_track(
                frame, concepts=concepts, persist=True, conf=conf
            )
            inference_seconds += time.perf_counter() - started
            detections.extend(frame_detections)
            for detection in frame_detections:
                if detection.get("track_id") is not None:
                    track_hits[detection["track_id"]] += 1
    finally:
        processor.release()
        detector.reset_tracking()

    stable_tracks = sum(1 for hits in track_hits.values() if hits >= 3)
    return {
        "video": video_path,
        "sampled_frames": sampled,
        "detections": len(detections),
        "unique_tracks": len(track_hits),
        "stable_tracks": stable_tracks,
        "avg_inference_ms": (inference_seconds / sampled * 1000.0) if sampled else 0.0,
        "total_inference_seconds": inference_seconds,
        "boxes": [d["bbox"] for d in detections],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", help="CCTV video paths")
    parser.add_argument("--objects", nargs="+", default=["speaker", "computer", "monitor", "laptop", "keyboard", "phone"])
    parser.add_argument("--weights", default=YOLOE_MODEL_PATH)
    parser.add_argument("--annotations")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--conf", type=float, default=OPEN_VOCAB_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    annotations = {}
    if args.annotations:
        with open(args.annotations, "r", encoding="utf-8") as handle:
            annotations = json.load(handle)

    detector = YOLOEDetector(args.weights)
    report = {"weights": args.weights, "confidence": args.conf, "objects": {}}
    for requested_object in args.objects:
        query = normalize_object_request(requested_object)
        object_report = []
        for video in args.videos:
            result = benchmark_video(video, detector, query["concepts"] or [requested_object], args.stride, args.conf)
            ground_truth = annotations.get(os.path.basename(video), {}).get(requested_object)
            if ground_truth is not None:
                tp, fp, fn = score_boxes(result["boxes"], ground_truth)
                result.update({
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(tp + fn, 1),
                })
            else:
                result.update({"precision": None, "recall": None, "ground_truth": "not provided"})
            result.pop("boxes", None)
            object_report.append(result)
        report["objects"][requested_object] = object_report

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

