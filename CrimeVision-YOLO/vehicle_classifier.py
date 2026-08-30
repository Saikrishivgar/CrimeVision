import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
from PIL import Image
from collections import Counter

class VehicleClassifier:
    def __init__(self, model_name="Jordo23/vehicle-classifier", threshold=0.75):
        """
        Uses EfficientNet-B4 trained on VMMRdb for vehicle make/model recognition.
        """
        self.threshold = threshold
        self.model = None
        self.processor = None
        device_override = os.environ.get("CRIMEVISION_DEVICE")
        if device_override:
            self.device = torch.device(device_override)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Standardized Brands for Hybrid Search
        self.STANDARD_BRANDS = {
            "toyota", "hyundai", "honda", "maruti suzuki", "tata", "mahindra", 
            "kia", "bmw", "mercedes-benz", "audi", "ford", "volkswagen", "chevrolet",
            "nissan", "renault", "skoda", "jeep", "mg", "volvo", "lexus", "porsche", "land rover"
        }
        
        try:
            self.processor = AutoImageProcessor.from_pretrained("dima806/car_models_image_detection")
            self.model = AutoModelForImageClassification.from_pretrained("dima806/car_models_image_detection")
            self.model.to(self.device)
            self.model.eval()
            print("Successfully loaded dima806/car_models_image_detection")
        except Exception as e:
            print(f"Warning: Failed to load HF vehicle model: {e}")
            self.model = None
            self.processor = None

    def map_brand(self, raw_make):
        raw_make = raw_make.lower().strip()
        if raw_make in self.STANDARD_BRANDS:
            return raw_make.capitalize()
        # Handle some common synonyms/typos
        if "mercedes" in raw_make:
            return "Mercedes-Benz"
        if "maruti" in raw_make:
            return "Maruti Suzuki"
        if "vw" in raw_make:
            return "Volkswagen"
        if "land-rover" in raw_make or "rover" in raw_make:
            return "Land Rover"
        return "Unknown"

    def classify_crop(self, image_crop):
        """
        Runs the classifier on a single crop.
        Returns: (brand, model, confidence)
        """
        if self.model is None or image_crop is None or image_crop.size == 0:
            return "Unknown", "Unknown", 0.0
            
        try:
            rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                
            prob = torch.nn.functional.softmax(logits, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            
            confidence = top_p.item()
            label = self.model.config.id2label[top_class.item()]
            
            # Model usually outputs "Make Model Year" or "Make Model"
            parts = label.split(" ")
            raw_make = parts[0]
            raw_model = " ".join(parts[1:]) if len(parts) > 1 else "Unknown"
            
            if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 4:
                raw_model = " ".join(parts[1:-1])
                
            return raw_make, raw_model, confidence
            
        except Exception as e:
            print(f"Classification error: {e}")
            return "Unknown", "Unknown", 0.0

    def classify_track(self, crops):
        """
        Takes multiple crops from the same tracked vehicle,
        runs classification, and votes for the best result.
        Returns: (brand, model, brand_confidence, model_confidence)
        """
        if not crops:
            return "Unknown", "Unknown", 0.0, 0.0
            
        results = []
        for crop in crops:
            make, model, conf = self.classify_crop(crop)
            results.append((make, model, conf))
            
        # Vote for make
        makes = [r[0] for r in results]
        make_counts = Counter(makes)
        best_make = make_counts.most_common(1)[0][0]
        
        # Calculate average confidence for the winning make
        make_confs = [r[2] for r in results if r[0] == best_make]
        make_conf = sum(make_confs) / len(make_confs) if make_confs else 0.0
        
        # Vote for model (only among predictions that got the winning make right)
        models = [r[1] for r in results if r[0] == best_make]
        if not models:
            models = [r[1] for r in results]
            
        model_counts = Counter(models)
        best_model = model_counts.most_common(1)[0][0]
        
        # Calculate average confidence for the winning model
        model_confs = [r[2] for r in results if r[1] == best_model]
        model_conf = sum(model_confs) / len(model_confs) if model_confs else 0.0
        
        # Map brand
        mapped_brand = self.map_brand(best_make)
        
        # Apply Thresholds
        if make_conf < self.threshold:
            mapped_brand = "Unknown"
            
        if model_conf < self.threshold:
            best_model = "Unknown"
            
        # If the make is generic/unknown but we got a high confidence hit for something else, 
        # we still suppress it to avoid hallucination.
        return mapped_brand, best_model, make_conf, model_conf
