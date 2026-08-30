import cv2
import time
from ultralytics import YOLO

video_path = "/Users/saikrishivgars/Desktop/YOLOO/CrimeVision-backend/storage/videos/video_71069654/Screen Recording 2026-08-08 at 6.40.13 PM.mov"
cap = cv2.VideoCapture(video_path)
print("Opened:", cap.isOpened())

model = YOLO("yolo11n.pt")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Done reading.")
        break
    count += 1
    if count % 10 == 0:
        t0 = time.time()
        model.track(frame, persist=True, device="mps", verbose=False)
        print(f"Frame {count} processed in {time.time() - t0:.2f}s")
