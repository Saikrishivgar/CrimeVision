import os
import glob
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Configuration
UCF_CRIME_DATASET_PATH = os.environ.get("UCF_CRIME_DATASET_PATH", "../archive (1)")
UCF_MAX_TRAIN_IMAGES_PER_CLASS = int(os.environ.get("UCF_MAX_TRAIN_IMAGES_PER_CLASS", 1000)) # Configurable subset
UCF_MAX_TEST_IMAGES_PER_CLASS = int(os.environ.get("UCF_MAX_TEST_IMAGES_PER_CLASS", 200))
MODEL_SAVE_DIR = "../models/ucf_crime"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, split="Train", max_per_class=None, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            files = glob.glob(os.path.join(cls_dir, "*.png"))
            if max_per_class is not None and len(files) > max_per_class:
                random.shuffle(files)
                files = files[:max_per_class]
            for f in files:
                self.samples.append((f, self.class_to_idx[cls_name]))
                
        print(f"Loaded {len(self.samples)} {split} samples across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("Initializing datasets...")
    train_dataset = UCFCrimeDataset(UCF_CRIME_DATASET_PATH, split="Train", max_per_class=UCF_MAX_TRAIN_IMAGES_PER_CLASS, transform=train_transform)
    val_dataset = UCFCrimeDataset(UCF_CRIME_DATASET_PATH, split="Test", max_per_class=UCF_MAX_TEST_IMAGES_PER_CLASS, transform=val_transform)
    
    # Save class names
    with open(os.path.join(MODEL_SAVE_DIR, "class_names.json"), "w") as f:
        json.dump(train_dataset.classes, f, indent=4)
        
    with open(os.path.join(MODEL_SAVE_DIR, "config.json"), "w") as f:
        json.dump({"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "input_size": 224}, f, indent=4)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model Setup (ResNet50)
    print("Loading pretrained ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best.pt"))
            print("  --> Saved new best model")

    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_model()
