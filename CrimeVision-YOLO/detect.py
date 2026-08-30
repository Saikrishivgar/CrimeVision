import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLO object detection on an image, directory, or video.")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="Path to model weights (defaults to best.pt if trained, else falls back to yolo11s.pt if not found)")
    parser.add_argument("--source", type=str, required=True, help="Path to image, directory of images, video, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--save", action="store_true", default=True, help="Save detection results (bounding boxes drawn)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu, mps, 0)")
    args = parser.parse_args()

    # Fallback checking
    if not os.path.exists(args.model) and args.model == "runs/detect/train/weights/best.pt":
        print(f"Warning: Custom weights '{args.model}' not found. Falling back to pretrained 'yolo11s.pt' for testing.")
        args.model = "yolo11s.pt"

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    # Convert '0' to int for webcam
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"Running prediction on: {source} with conf={args.conf}...")
    
    # Run predict
    results = model.predict(
        source=source,
        conf=args.conf,
        save=args.save,
        device=args.device
    )
    print("Inference completed!")
    if args.save:
        print("Results saved in the default 'runs/' directory.")

if __name__ == "__main__":
    main()
