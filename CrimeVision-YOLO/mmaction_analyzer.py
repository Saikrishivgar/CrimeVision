import torch
import cv2
import numpy as np

class TemporalActionAnalyzer:
    def __init__(self, model_name="MCG-NJU/videomae-base-finetuned-kinetics"):
        """
        Initializes the temporal action model. 
        If MMAction2 is not installed, it acts as a fallback temporal-action model using VideoMAE.
        It does NOT assume it can natively detect 'robbery' or 'accident'. It outputs generic actions (e.g., 'falling', 'running', 'fighting').
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.is_loaded = False
        
        try:
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("Temporal action model fallback (VideoMAE) loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load temporal action model: {e}")

    def analyze_clip(self, video_path, start_time, end_time):
        """
        Extracts a clip from the video file between start_time and end_time,
        and returns a structured analysis.
        Args:
            video_path: str, path to the video
            start_time: float, start time in seconds
            end_time: float, end time in seconds
        Returns:
            { "actions": [{"label": "running", "score": 0.81}, ...] }
        """
        if not self.is_loaded:
            return {"actions": []}
            
        try:
            # Critical fix for Apple Silicon macOS thread deadlocks
            cv2.setNumThreads(0)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"actions": []}
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
            cap.release()
            
            if len(frames) < 16:
                return {"actions": []}
            
            # Subsample 16 frames uniformly
            idx = np.linspace(0, len(frames) - 1, 16).astype(int)
            clip = [cv2.resize(frames[i], (224, 224)) for i in idx]
            
            inputs = self.processor(list(clip), return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                
            top_p, top_class = probs.topk(3, dim=1)
            
            results = []
            for i in range(3):
                score = top_p[0][i].item()
                label = self.model.config.id2label[top_class[0][i].item()]
                results.append({"label": label.lower(), "score": score})
                
            return {"actions": results}
        except Exception as e:
            print(f"Error during temporal action analysis: {e}")
            return {"actions": []}
