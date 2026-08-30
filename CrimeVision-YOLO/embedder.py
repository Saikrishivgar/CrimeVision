import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import numpy as np

class Embedder:
    """
    Generates semantic vector embeddings for both image crops and text queries using CLIP.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading CLIP embedding model ({model_name})...")
        # Force CPU to avoid PyTorch MPS thread deadlocks in FastAPI
        self.device = "cpu"
            
        print(f"CLIP device: {self.device}")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            self.model = None
            self.processor = None

    def embed_crop(self, crop):
        """
        Takes a cv2 BGR image crop, passes it through CLIP, and returns a 512-dim numpy vector.
        """
        if crop is None or crop.size == 0 or self.model is None or self.processor is None:
            return None
            
        try:
            # Convert BGR (OpenCV) to RGB and then PIL Image
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Normalize vector
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Convert back to cpu numpy array for FAISS
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error embedding crop: {e}")
            return None

    def embed_text(self, text):
        """
        Takes a natural language string, passes it through CLIP, and returns a 512-dim numpy vector.
        """
        if not text or self.model is None or self.processor is None:
            return None
            
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
            # Normalize vector
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error embedding text: {e}")
            return None
