"""Lazy YOLOE adapter.

This module must not import or initialize YOLOE during normal YOLO11 startup.
The class is instantiated only after an explicit arbitrary-object request.
"""

import os


class YOLOEDetector:
    source_model = "yoloe"

    def __init__(self, model_path, device=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLOE checkpoint not found at {model_path}. "
                "Download it only after the implementation is ready."
            )

        from ultralytics import YOLOE

        self.model_path = model_path
        self.device = device
        self.model = YOLOE(model_path, verbose=False)
        self._prompt_key = None
        self._concepts = []

    def set_concepts(self, concepts):
        concepts = [str(c).strip().lower() for c in concepts if str(c).strip()]
        concepts = list(dict.fromkeys(concepts))
        if not concepts:
            raise ValueError("At least one YOLOE object concept is required")

        prompt_key = tuple(concepts)
        if prompt_key != self._prompt_key:
            self.model.set_classes(concepts)
            self._prompt_key = prompt_key
            self._concepts = concepts
        return concepts

    def detect_and_track(
        self,
        frame,
        concepts,
        persist=True,
        conf=0.35,
        tracker="botsort.yaml",
    ):
        concepts = self.set_concepts(concepts)
        kwargs = {
            "persist": persist,
            "conf": conf,
            "tracker": tracker,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model.track(frame, **kwargs)
        result = results[0]
        detections = []
        if result.boxes is None:
            return detections

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())

            if isinstance(names, dict):
                prompt_class = names.get(cls_id, concepts[min(cls_id, len(concepts) - 1)])
            else:
                prompt_class = names[cls_id]

            detections.append(
                {
                    "bbox": tuple(int(v) for v in xyxy),
                    "class": str(prompt_class),
                    "confidence": float(box.conf[0]),
                    "track_id": track_id,
                    "source_model": self.source_model,
                    "query_concepts": concepts,
                }
            )
        return detections

    def reset_tracking(self):
        """Release Ultralytics' per-stream tracker state after a scan."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            # Ultralytics reuses an existing ``trackers`` attribute when
            # persist=True.  An empty list therefore causes the next scan to
            # fail at ``trackers[0]`` instead of reinitializing BoT-SORT.
            # Removing the attribute lets its normal callback create a fresh
            # tracker for the next query/video.
            if hasattr(predictor, "trackers"):
                del predictor.trackers
            if hasattr(predictor, "vid_path"):
                del predictor.vid_path
