import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class UCFCrimeSpatialModel:
    def __init__(self, model_dir="models/ucf_crime"):
        self.model_dir = model_dir
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.classes = []
        self.transform = None
        self._load_model()
        
    def _load_model(self):
        config_path = os.path.join(self.model_dir, "config.json")
        classes_path = os.path.join(self.model_dir, "class_names.json")
        weights_path = os.path.join(self.model_dir, "best.pt")
        
        if not os.path.exists(weights_path) or not os.path.exists(classes_path):
            print(f"Warning: UCF Crime Spatial model not found in {self.model_dir}")
            return
            
        with open(classes_path, "r") as f:
            self.classes = json.load(f)
            
        mean, std, input_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                mean = config.get("mean", mean)
                std = config.get("std", std)
                input_size = config.get("input_size", input_size)
                
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Load ResNet50 Architecture
        self.model = models.resnet50()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.classes))
        
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Loaded UCFCrimeSpatialModel with {len(self.classes)} classes.")

    def is_loaded(self):
        return self.model is not None

    def predict_frame(self, image_path):
        if not self.is_loaded():
            return None
            
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            class_probs = {self.classes[i]: probs[i].item() for i in range(len(self.classes))}
            return class_probs
        except Exception as e:
            print(f"Error in UCF Crime prediction: {e}")
            return None

    def predict_frames_averaged(self, frame_paths):
        if not self.is_loaded() or not frame_paths:
            return None
            
        all_probs = []
        for path in frame_paths:
            probs = self.predict_frame(path)
            if probs:
                all_probs.append(probs)
                
        if not all_probs:
            return None
            
        avg_probs = {cls: 0.0 for cls in self.classes}
        for probs in all_probs:
            for cls, val in probs.items():
                avg_probs[cls] += val
                
        num_frames = len(all_probs)
        for cls in avg_probs:
            avg_probs[cls] /= num_frames
            
        return avg_probs
