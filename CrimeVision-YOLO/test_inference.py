from inference import run_pipeline
import os
YOLO_DIR = "/Users/saikrishivgars/Desktop/YOLOO/CrimeVision-YOLO"
model_path = os.path.join(YOLO_DIR, "yolo11s.pt")
source = "/Users/saikrishivgars/Desktop/YOLOO/CrimeVision-backend/storage/videos/video_e668aa2b/WhatsApp Video 2026-08-03 at 22.38.47.mp4"
try:
    res = run_pipeline(source=source, model_path=model_path, conf=0.15, save_video=None, show=False)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
