import os
import cv2
import json
import subprocess
from clip_generator import generate_clip
from database import get_event_by_id

def get_or_create_evidence(event_id: str, storage_dir: str):
    """
    Retrieves or generates evidence for a given event ID.
    Returns a dict with clip_url, frame_url, and event metadata.
    """
    event = get_event_by_id(event_id)
    if not event:
        return None
        
    video_name = event.get("video_name")
    video_dir = os.path.join(storage_dir, "videos", event["video_id"])
    
    if not video_name:
        if os.path.exists(video_dir):
            files = [f for f in os.listdir(video_dir) if not f.startswith(".")]
            if files:
                video_name = files[0]
                
    if not video_name:
        return None
        
    video_path = os.path.join(video_dir, video_name)
    if not os.path.exists(video_path):
        return None
        
    evidence_dir = os.path.join(storage_dir, "evidence")
    frames_dir = os.path.join(evidence_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    clip_filename = f"{event_id}.mp4"
    frame_filename = f"{event_id}.jpg"
    
    clip_path = os.path.join(evidence_dir, clip_filename)
    frame_path = os.path.join(frames_dir, frame_filename)
    
    # Generate Clip
    if not os.path.exists(clip_path):
        # Fallback defaults if clip_start/end are not perfectly valid
        c_start = max(0, float(event.get("clip_start", event["timestamp"] - 3)))
        c_end = float(event.get("clip_end", event["timestamp"] + 3))
        generate_clip(video_path, c_start, c_end, clip_path)
        
    # Generate Highlighted Frame
    if not os.path.exists(frame_path):
        cap = cv2.VideoCapture(video_path)
        # Attempt to jump to the exact frame, fallback to timestamp
        frame_idx = event.get("frame")
        if frame_idx is not None and frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, event["timestamp"] * 1000)
            
        ret, frame_img = cap.read()
        if ret:
            bbox = event.get("bbox", [])
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                # Draw bounding box (Blue-ish color)
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 165, 255), 3)
                
                # Draw label
                label = event.get("object", "Object").upper()
                # Optional color prepended
                color_attr = event.get("attributes", {}).get("vehicle_color")
                if not color_attr and event.get("attributes", {}).get("shirt_color"):
                    color_attr = event.get("attributes", {}).get("shirt_color")
                    
                if color_attr and color_attr != "Unknown":
                    label = f"{color_attr} {label}".upper()
                    
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame_img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 165, 255), -1)
                cv2.putText(frame_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            cv2.imwrite(frame_path, frame_img)
        cap.release()
        
    return {
        "clip_url": f"/storage/evidence/{clip_filename}",
        "frame_url": f"/storage/evidence/frames/{frame_filename}",
        "event": event
    }
