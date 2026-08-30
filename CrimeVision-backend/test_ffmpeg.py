import cv2
import threading

def read_video():
    video_path = "/Users/saikrishivgars/Desktop/YOLOO/CrimeVision-backend/storage/videos/video_71069654/Screen Recording 2026-08-08 at 6.40.13 PM.mov"
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    print("Opened with FFMPEG:", cap.isOpened())
    if cap.isOpened():
        ret, frame = cap.read()
        print("Read frame:", ret)

t = threading.Thread(target=read_video)
t.start()
t.join()
