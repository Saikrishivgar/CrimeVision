import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model on the CrimeVision dataset.")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Pretrained model name or path (e.g., yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu, mps, 0, or auto-selected if not specified)")
    args = parser.parse_args()

    print(f"Initializing YOLO model with weights: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training on data: {args.data} for {args.epochs} epochs...")
    
    # Train the model
    # Note: Ultralytics automatically handles device configuration if device is set to None.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name="train"
    )
    print("Training completed! Best weights saved to runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
