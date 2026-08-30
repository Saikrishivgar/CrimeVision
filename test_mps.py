from ultralytics import YOLO
import torch
import cv2

print("MPS available:", torch.backends.mps.is_available())
model = YOLO("yolo11n.pt")
frame = cv2.imread("CrimeVision-YOLO/dataset/test.jpg") # use a random dummy frame
if frame is None:
    frame = torch.zeros((480, 640, 3), dtype=torch.uint8).numpy()
    
print("Running inference on MPS...")
try:
    res = model.track(frame, device="mps", persist=True)
    print("Success!")
except Exception as e:
    print("Error:", e)
