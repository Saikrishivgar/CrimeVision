import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import cv2
import numpy as np

class AttributeExtractor:
    """
    Extracts visual attributes (like colors) from cropped image regions using microsoft/Florence-2-base.
    """
    def __init__(self):
        print("Loading Florence-2 model for visual attribute extraction...")
        # Force CPU to avoid PyTorch MPS thread deadlocks in FastAPI
        self.device = "cpu"
            
        print(f"Florence-2 running on device: {self.device}")
        
        # Load processor and model with fallback
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Florence-2 failed to load ({e}). Color extraction disabled.")
            self.model = None
            self.processor = None
        
        # Predefined list of standard colors for parsing
        self.standard_colors = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink", "Black", "White", "Gray", "Brown"]

    def describe_crop(self, crop, prompt="<CAPTION>"):
        """
        Runs Florence-2 on a cv2 image crop and returns the text description.
        """
        if crop is None or crop.size == 0 or self.model is None or self.processor is None:
            return ""
            
        try:
            # Convert BGR (OpenCV) to RGB and then PIL Image
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt"
            ).to(self.device)
            
            # Florence-2 output generation
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128
            )
            
            result = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            return result
            
        except Exception as e:
            print(f"Florence-2 inference failed: {e}")
            return ""

    def extract_color_from_text(self, text, default="Unknown"):
        """
        Parses text to find standard colors.
        """
        if not text:
            return default
            
        words = text.lower().replace(",", " ").replace(".", " ").split()
        for w in words:
            if w == "grey":
                return "Gray"
            for c in self.standard_colors:
                if w == c.lower():
                    return c
        return default

    def extract_person_attributes(self, frame, bbox):
        """
        Extracts shirt color and pant color for a detected person.
        
        Args:
            frame: Full BGR frame.
            bbox: (x1, y1, x2, y2) bounding box of the person.
        """
        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        
        # Keep inside image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {"shirt_color": "Unknown", "pant_color": "Unknown"}
            
        ch, cw, _ = crop.shape
        
        # Crop the shirt (upper body heuristic: 10% to 45% of height, center 15% to 85% width)
        shirt_y1 = int(0.10 * ch)
        shirt_y2 = int(0.45 * ch)
        shirt_x1 = int(0.15 * cw)
        shirt_x2 = int(0.85 * cw)
        
        shirt_crop = crop[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
        shirt_desc = self.describe_crop(shirt_crop)
        shirt_color = self.extract_color_from_text(shirt_desc, "Unknown")
        
        # Crop the pants (lower body heuristic: 55% to 90% of height, center 15% to 85% width)
        pant_y1 = int(0.55 * ch)
        pant_y2 = int(0.90 * ch)
        pant_x1 = int(0.15 * cw)
        pant_x2 = int(0.85 * cw)
        
        pant_crop = crop[pant_y1:pant_y2, pant_x1:pant_x2]
        pant_desc = self.describe_crop(pant_crop)
        pant_color = self.extract_color_from_text(pant_desc, "Unknown")
        
        return {
            "shirt_color": shirt_color,
            "pant_color": pant_color
        }

    def extract_vehicle_attributes(self, frame, bbox):
        """
        Extracts dominant color of a vehicle crop.
        
        Args:
            frame: Full BGR frame.
            bbox: (x1, y1, x2, y2) bounding box of the vehicle.
        """
        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        
        # Keep inside image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {"vehicle_color": "Unknown"}
            
        desc = self.describe_crop(crop)
        vehicle_color = self.extract_color_from_text(desc, "Unknown")
        
        # Fallback: if color is Unknown, use a center-crop to be safe
        if vehicle_color == "Unknown":
            ch, cw, _ = crop.shape
            veh_y1 = int(0.35 * ch)
            veh_y2 = int(0.75 * ch)
            veh_x1 = int(0.15 * cw)
            veh_x2 = int(0.85 * cw)
            veh_crop = crop[veh_y1:veh_y2, veh_x1:veh_x2]
            desc_center = self.describe_crop(veh_crop)
            vehicle_color = self.extract_color_from_text(desc_center, "Unknown")
        
        return {
            "vehicle_color": vehicle_color
        }
