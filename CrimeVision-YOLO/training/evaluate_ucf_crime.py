import os
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from train_ucf_crime import UCFCrimeDataset, UCF_MAX_TEST_IMAGES_PER_CLASS, BATCH_SIZE

MODEL_SAVE_DIR = "../models/ucf_crime"
UCF_CRIME_DATASET_PATH = os.environ.get("UCF_CRIME_DATASET_PATH", "../archive (1)")

def evaluate_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    config_path = os.path.join(MODEL_SAVE_DIR, "config.json")
    classes_path = os.path.join(MODEL_SAVE_DIR, "class_names.json")
    weights_path = os.path.join(MODEL_SAVE_DIR, "best.pt")
    
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        return
        
    with open(classes_path, "r") as f:
        classes = json.load(f)
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    val_transform = transforms.Compose([
        transforms.Resize((config["input_size"], config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["mean"], std=config["std"])
    ])
    
    print("Loading test dataset...")
    val_dataset = UCFCrimeDataset(UCF_CRIME_DATASET_PATH, split="Test", max_per_class=UCF_MAX_TEST_IMAGES_PER_CLASS, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Loading model...")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n==================================================")
    print("EVALUATION RESULTS")
    print("==================================================")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\nClassification Report:\n")
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n")
    print(cm)

if __name__ == "__main__":
    evaluate_model()
