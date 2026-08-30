# CrimeVision YOLOE Test Procedure

YOLO11 remains the always-on detector. These checks exercise the optional YOLOE branch without re-running or deleting existing video data.

## Preflight

From the repository root:

```bash
python -m py_compile \
  CrimeVision-backend/database.py \
  CrimeVision-backend/main.py \
  CrimeVision-backend/object_query.py \
  CrimeVision-YOLO/extended_object_scan.py \
  CrimeVision-YOLO/detectors/yoloe_detector.py

cd frontend && npm run build
```

The backend should import with `yoloe_loaded False`; YOLOE must not be initialized during normal startup.

## Query-routing checks

```bash
python -c 'import sys; sys.path.insert(0, "CrimeVision-backend"); from object_query import normalize_object_request; [print(q, normalize_object_request(q)) for q in ["Find the speaker", "Find the PC cabinet", "Find the red car", "Fully map this video including speakers and computers"]]'
```

Expected behavior:

- speaker, PC cabinet, monitor, laptop, keyboard, phone and user-defined uncommon objects route to YOLOE.
- red car routes to YOLO11/BoT-SORT and does not initialize YOLOE.
- normal full mapping is YOLO11-only unless the Extended Object Detection control is ON or the request names arbitrary objects.

## Manual UI acceptance

With the backend and `frontend` running:

1. Keep Extended Object Detection OFF and request `Fully map this video`. Confirm the map contains existing YOLO11 tracks and no YOLOE scan status.
2. Select one video and request `Find the speaker`, `Find the black speaker`, `Find the PC cabinet`, `Show all monitors`, and `Fully map this video including speakers`. Confirm the visible YOLOE status, selected-video scoping, and completion behavior.
3. Turn Extended Object Detection ON and request a normal full map. Confirm query-scoped electronics scans run and the resulting objects appear in the same map.
4. Verify live boxes use stored point coordinates, labels show friendly IDs/confidence/source, trails follow points, stationary objects show `Stationary`, and Raw Data includes `track_uid`/source fields.
5. Repeat with `across all videos` and confirm results contain the requested videos only.
6. Run an existing accident/fight search and confirm YOLOE objects are not incident candidates.

## Benchmark

The benchmark reports sampled frames, raw detections, unique tracks, stable tracks, and inference time. Precision and recall are only emitted when manually reviewed annotations are supplied; confidence is not treated as ground truth.

```bash
cd CrimeVision-YOLO
python benchmark_yoloe.py \
  ../CrimeVision-backend/storage/videos/<video-id>/<file> \
  --objects speaker computer monitor laptop keyboard phone \
  --stride 10 \
  --annotations annotations.json
```

Annotation format:

```json
{
  "video.mp4": {
    "speaker": [[x1, y1, x2, y2]],
    "computer": [[x1, y1, x2, y2]]
  }
}
```

Review the priority classes in this order: speaker, computer/PC cabinet, monitor, laptop, keyboard, phone. Calibrate `CRIMEVISION_OPEN_VOCAB_CONFIDENCE_THRESHOLD`, minimum hits, maximum gap, and bounding-box stability from reviewed results before production use.

## Regression checks

Record the counts of `videos`, `tracks`, `track_points`, and `events` before and after additive schema migration. They must not decrease. Confirm FAISS/evidence generation, Gallery, archive/delete, multi-video selection, and the original YOLO11 → BoT-SORT → incident → VideoMAE path remain available.
