import os
import sys

sys.path.append(os.path.join(os.getcwd(), "CrimeVision-YOLO"))
from inference import run_pipeline
from video_processor import VideoProcessor

if __name__ == "__main__":
    test_video = "test.mp4"
    if not os.path.exists(test_video):
        print(f"Error: {test_video} not found")
        sys.exit(1)
        
    print(f"Running inference on {test_video}...")
    try:

        processor = VideoProcessor(test_video)
        results = run_pipeline(test_video)
        print("Inference completed successfully!")

        # Verify incidents
        incidents = [r for r in results if r.get("event_id")]
        print(f"Detected {len(incidents)} incidents.")
        for inc in incidents:
            print(f" - {inc.get('event_type')} at {inc.get('timestamp')}s (confidence: {inc.get('confidence')})")
            if "evidence" in inc.get("attributes", {}):
                print(f"   Evidence: {inc['attributes']['evidence']}")
                
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
