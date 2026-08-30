import os
from ultralytics import YOLO

class YOLODetector:
    """
    Wraps Ultralytics YOLO logic for object detection and tracking.
    Uses built-in ByteTrack/BoT-SORT tracking through the `.track()` API.
    """
    def __init__(self, model_path="runs/detect/train/weights/best.pt", fallback_model="yolo11n.pt"):
        if not os.path.exists(model_path):
            print(f"Warning: Custom weights '{model_path}' not found. Falling back to pretrained '{fallback_model}'...")
            self.model = YOLO(fallback_model)
        else:
            print(f"Loading YOLO weights from: {model_path}")
            self.model = YOLO(model_path)

    def detect_and_track(self, frame, persist=True, conf=0.25, tracker="botsort.yaml"):
        """
        Runs object detection and tracking on a frame.
        
        Args:
            frame: OpenCV image frame (numpy array).
            persist: bool, whether to maintain tracking state across frames.
            conf: float, confidence threshold for detections.
            tracker: string, tracking configuration file.
            
        Returns:
            list of dicts containing bbox, class, confidence, track_id
        """
        import torch
        # Use env override for cloud deployment (e.g. CRIMEVISION_DEVICE=cpu on Render)
        device_override = os.environ.get("CRIMEVISION_DEVICE")
        if device_override:
            device = device_override
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        # Apple Silicon MPS can sometimes deadlock with ultralytics multiprocessing,
        # but the user requested MPS for speed. We'll use MPS and if it hangs, it's a known PyTorch issue.
        
        if persist:
            results = self.model.track(frame, persist=True, conf=conf, tracker=tracker, verbose=False, device=device)
        else:
            results = self.model(frame, conf=conf, verbose=False, device=device)

        detections = []
        res = results[0]
        
        if res.boxes is not None:
            boxes = res.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                # Extract tracking ID if it exists
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": class_name,
                    "confidence": conf_score,
                    "track_id": track_id
                })
                
        return detections
