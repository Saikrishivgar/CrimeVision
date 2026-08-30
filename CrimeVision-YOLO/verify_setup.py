import cv2
import numpy as np
import subprocess
import os
import json

def create_dummy_video(filename="dummy_test.mp4", duration=2, fps=10):
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        # Create black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a shape resembling a vehicle or bounding area to trigger detection/inference paths
        cv2.rectangle(frame, (200, 200), (440, 380), (100, 150, 200), -1)
        # Draw a person shape (e.g. circle for head, rectangle for body)
        cv2.circle(frame, (100, 100), 30, (0, 0, 255), -1)
        cv2.rectangle(frame, (80, 130), (120, 250), (255, 0, 0), -1)
        
        out.write(frame)
        
    out.release()
    print(f"Created dummy video: {filename}")

def main():
    dummy_video = "dummy_test.mp4"
    output_json = "test_output.json"
    
    # 1. Create dummy video
    create_dummy_video(dummy_video)
    
    # 2. Run the inference pipeline
    print("Running pipeline inference on dummy video...")
    try:
        result = subprocess.run([
            "python3", "inference.py",
            "--model", "yolo11s.pt",
            "--source", dummy_video,
            "--save-json", output_json
        ], capture_output=True, text=True, check=True)
        print("Pipeline execution stdout:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Pipeline execution failed!")
        print("Stderr:\n", e.stderr)
        print("Stdout:\n", e.stdout)
        return

    # 3. Verify JSON output
    if not os.path.exists(output_json):
        print(f"Error: {output_json} was not created.")
        return
        
    try:
        with open(output_json, "r") as f:
            data = json.load(f)
            
        print("\nVerifying JSON Structure:")
        print(json.dumps(data, indent=2))
        
        assert "video_name" in data, "Missing 'video_name'"
        assert "duration" in data, "Missing 'duration'"
        assert "detections" in data, "Missing 'detections'"
        
        print("\nSuccess! JSON schema verified successfully.")
        
    except Exception as e:
        print(f"Verification check failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(dummy_video):
            os.remove(dummy_video)
        if os.path.exists(output_json):
            os.remove(output_json)
        print("Temporary test files cleaned up.")

if __name__ == "__main__":
    main()
