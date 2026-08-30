import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_chat(query, video_id="test_vid_01"):
    payload = {
        "messages": [{"role": "user", "content": query}],
        "video_id": video_id
    }
    try:
        r = requests.post(f"{BASE_URL}/chat", json=payload)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Testing /chat endpoints to ensure they don't break or reload YOLO...")
    
    queries = [
        "Find the red car",
        "Show the evidence",
        "Track the blue car",
        "Did any accident happen?",
        "Show incident timeline"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        res = test_chat(q)
        print("Response text:", res.get("response", res.get("error", "No response")))
        
    print("\nRegression test calls completed.")
