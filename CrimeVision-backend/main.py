import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevent macOS segfault with FAISS + PyTorch
import torch # CRITICAL: MUST import torch BEFORE faiss to prevent OpenMP segfaults!
import sys
import shutil
import uuid
import json
import asyncio
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2

# Add YOLO engine path to import modules
YOLO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../CrimeVision-YOLO"))
if YOLO_DIR not in sys.path:
    sys.path.append(YOLO_DIR)

# Import custom modules for the AI Command Center
from database import (
    DB_PATH,
    init_db,
    insert_events,
    query_events,
    get_all_events,
    get_event_by_id,
    insert_tracks,
    get_video_tracks,
    get_all_videos_list,
    delete_video_data,
    set_video_archived,
    get_next_track_id,
    has_open_vocab_tracks,
    create_chat_session,
    insert_chat_message,
    get_chat_sessions,
    get_chat_messages,
    delete_chat_session,
)
from vector_store import VectorStore
from embedder import Embedder
from nlp_query import optimize_prompt, generate_fir_text
from object_query import normalize_object_request, extract_object_requests, ELECTRONIC_OBJECTS
from detectors.detector_router import should_use_yoloe
from clip_generator import generate_clip
from evidence_manager import get_or_create_evidence

try:
    from inference import run_pipeline
except ImportError as e:
    print(f"Error importing from CrimeVision-YOLO: {e}")
    raise e

app = FastAPI(title="CrimeVision AI Video Analytics Backend")

DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "storage"))
for sub in ["videos", "frames", "clips", "metadata", "faiss", "evidence", "evidence/frames"]:
    os.makedirs(os.path.join(STORAGE_DIR, sub), exist_ok=True)

app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

vector_store = None
text_embedder = None
analysis_status = {}
extended_scan_status = {}
model_registry = {}

@app.on_event("startup")
def on_startup():
    global vector_store, text_embedder, model_registry
    print("Initializing Database...")
    init_db()
    print("Initializing FAISS Vector Store...")
    vector_store = VectorStore(
        index_path=os.path.join(STORAGE_DIR, "faiss", "crimevision.index"),
        meta_path=os.path.join(STORAGE_DIR, "faiss", "crimevision_meta.pkl")
    )
    print("Initializing CLIP Text Embedder...")
    text_embedder = Embedder()
    model_registry["embedder"] = text_embedder
    
    print("Initializing YOLO Model...")
    from detector import YOLODetector
    model_registry["detector"] = YOLODetector(os.path.join(YOLO_DIR, "yolo11n.pt"))
    
    print("Initializing Vehicle Classifier...")
    from vehicle_classifier import VehicleClassifier
    model_registry["vehicle_classifier"] = VehicleClassifier()
    
    print("Initializing Temporal Action Analyzer...")
    from mmaction_analyzer import TemporalActionAnalyzer
    model_registry["temporal_analyzer"] = TemporalActionAnalyzer()
    
    print("Initializing Florence-2 Attribute Extractor...")
    from attribute import AttributeExtractor
    model_registry["attribute_extractor"] = AttributeExtractor()


def _video_path(video_id):
    video_dir = os.path.join(STORAGE_DIR, "videos", video_id)
    if not os.path.exists(video_dir):
        return None
    files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]
    return os.path.join(video_dir, files[0]) if files else None


def _assign_track_identities(video_id, detections):
    """Assign stable public IDs while preserving detector-local IDs in metadata."""
    next_id = get_next_track_id(video_id)
    for detection in detections:
        source_model = detection.get("source_model") or "yolo11"
        detection["source_model"] = source_model
        track_id = detection.get("track_id")
        if track_id is None:
            continue
        if source_model == "yoloe" and track_id >= 0:
            detection["detector_track_id"] = track_id
            detection["track_id"] = next_id
            next_id += 1
        if detection.get("track_id") is not None and detection.get("track_id") != -1:
            detection["track_uid"] = f"{source_model}:{video_id}:{detection['track_id']}"
            for point in detection.get("points", []):
                point["track_id"] = detection["track_id"]
                point["track_uid"] = detection["track_uid"]
                point["source_model"] = source_model


def _persist_detection_results(video_id, results):
    detections = results.get("detections", []) if results else []
    _assign_track_identities(video_id, detections)
    for detection in detections:
        detection["video_id"] = video_id
        detection["camera_id"] = analysis_status.get(video_id, {}).get("camera_id", "CAM_01")
        detection["location"] = analysis_status.get(video_id, {}).get("location", "Unknown")
        detection.setdefault("source_model", "yolo11")
        embedding = detection.pop("embedding", None)
        if embedding is not None and vector_store is not None:
            vector_store.add_embedding(
                embedding=embedding,
                event_id=detection.get("event_id", str(uuid.uuid4())),
            )

    if detections:
        insert_events(detections)
        insert_tracks(video_id, detections)
        if vector_store is not None:
            vector_store.save()
    return detections


def _find_existing_extended_tracks(video_id, object_names):
    tracks = get_video_tracks(video_id)
    wanted = {name.lower() for name in object_names}
    return [
        track for track in tracks
        if track.get("source_model") == "yoloe"
        and track.get("class_name", "").lower() in wanted
    ]

@app.get("/videos")
async def get_videos():
    db_videos = get_all_videos_list(STORAGE_DIR)
    video_dict = {v["video_id"]: v for v in db_videos}
    
    for vid, status in analysis_status.items():
        if vid in video_dict:
            if video_dict[vid]["filename"] == "unknown.mp4":
                video_dict[vid]["filename"] = status.get("filename", "unknown.mp4")
            video_dict[vid]["status"] = status.get("status", "READY")
        else:
            video_dict[vid] = {
                "video_id": vid,
                "filename": status.get("filename", "unknown.mp4"),
                "duration": 0,
                "status": status.get("status", "processing")
            }
            
    return list(video_dict.values())

@app.get("/videos/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: str):
    thumbnails_dir = os.path.join(STORAGE_DIR, "thumbnails")
    os.makedirs(thumbnails_dir, exist_ok=True)
    thumb_path = os.path.join(thumbnails_dir, f"{video_id}.jpg")
    
    if os.path.exists(thumb_path):
        return FileResponse(thumb_path)
        
    # Generate thumbnail if it doesn't exist
    video_dir = os.path.join(STORAGE_DIR, "videos", video_id)
    if os.path.exists(video_dir):
        files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        if files:
            vid_path = os.path.join(video_dir, files[0])
            cap = cv2.VideoCapture(vid_path)
            if cap.isOpened():
                # Read first frame
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(thumb_path, frame)
                    cap.release()
                    return FileResponse(thumb_path)
            cap.release()
            
    raise HTTPException(status_code=404, detail="Thumbnail could not be generated.")

@app.post("/videos/{video_id}/archive")
async def archive_video(video_id: str, payload: dict):
    is_archived = payload.get("is_archived", True)
    set_video_archived(video_id, is_archived)
    return {"status": "success", "video_id": video_id, "is_archived": is_archived}

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    # 1. Delete from DB and get orphaned event IDs
    event_ids = delete_video_data(video_id)
    
    # 2. Delete from FAISS safely
    if vector_store:
        vector_store.delete_by_event_ids(event_ids)
        
    # 3. Clean up evidence files
    evidence_dir = os.path.join(STORAGE_DIR, "evidence")
    frames_dir = os.path.join(evidence_dir, "frames")
    
    for evt_id in event_ids:
        clip_path = os.path.join(evidence_dir, f"{evt_id}.mp4")
        frame_path = os.path.join(frames_dir, f"{evt_id}.jpg")
        if os.path.exists(clip_path):
            os.remove(clip_path)
        if os.path.exists(frame_path):
            os.remove(frame_path)
            
    # 4. Remove Thumbnail
    thumb_path = os.path.join(STORAGE_DIR, "thumbnails", f"{video_id}.jpg")
    if os.path.exists(thumb_path):
        os.remove(thumb_path)
        
    # 5. Remove Video Directory
    video_dir = os.path.join(STORAGE_DIR, "videos", video_id)
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
        
    return {"status": "success", "message": f"Video {video_id} and all associated data permanently deleted."}

@app.get("/videos/{video_id}/map")
async def get_video_map_data(video_id: str):
    import sqlite3
    tracks = get_video_tracks(video_id)
    events = get_all_events(video_id)
    
    if not tracks and not events:
        raise HTTPException(status_code=404, detail="No track or event data found for this video.")
        
    if not tracks:
        # Fallback to events
        track_groups = {}
        for e in events:
            tid = e.get("track_id", -1)
            if tid == -1:
                tid = e.get("event_id", "unknown")
            if tid not in track_groups:
                track_groups[tid] = []
            track_groups[tid].append(e)
            
        tracks = []
        for tid, evts in track_groups.items():
            obj_class = evts[0].get("object", "unknown").lower()
            attrs = evts[0].get("attributes", {})
            if isinstance(attrs, str):
                try: attrs = json.loads(attrs)
                except: attrs = {}
            tracks.append({
                "track_id": tid,
                "track_uid": evts[0].get("track_uid") or f"{evts[0].get('source_model', 'yolo11')}:{video_id}:{tid}",
                "class_name": obj_class,
                "color": attrs.get("vehicle_color") or attrs.get("shirt_color") or "Unknown",
                "brand": attrs.get("vehicle_make", "Unknown"),
                "model": attrs.get("vehicle_model", "Unknown"),
                "source_model": evts[0].get("source_model", "yolo11"),
                "query_concepts": evts[0].get("query_concepts", []),
                "stationary": bool(evts[0].get("stationary", False)),
                "first_seen": min(e.get("timestamp", 0) for e in evts),
                "last_seen": max(e.get("timestamp", 0) for e in evts),
                "description": evts[0].get("description", ""),
                "points": []
            })
            
    # Get filename
    filename = "unknown.mp4"
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT filename FROM videos WHERE video_id = ?", (video_id,))
    row = c.fetchone()
    if row:
        filename = row["filename"]
    conn.close()
    
    video_meta = next((v for v in get_all_videos_list(STORAGE_DIR) if v["video_id"] == video_id), {})
    return {
        "video_id": video_id,
        "filename": filename,
        "duration": video_meta.get("duration", max((t.get("last_seen", 0) for t in tracks), default=0)),
        "fps": video_meta.get("fps", 24.0),
        "frame_count": video_meta.get("frame_count", 0),
        "tracks": tracks,
        "events": events,
    }

@app.post("/videos/upload")
async def upload_video(file: UploadFile = File(...), camera_id: str = Form("CAM_01"), location: str = Form("Main Gate")):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail="Only video files are supported.")
        
    unique_id = f"video_{uuid.uuid4().hex[:8]}"
    video_dir = os.path.join(STORAGE_DIR, "videos", unique_id)
    os.makedirs(video_dir, exist_ok=True)
    
    filename = file.filename
    filepath = os.path.join(video_dir, filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    analysis_status[unique_id] = {
        "status": "uploaded",
        "progress": 0,
        "stages": ["✓ Video uploaded"],
        "filename": filename,
        "camera_id": camera_id,
        "location": location
    }
    
    return {"video_id": unique_id, "filename": filename}

@app.post("/videos/{video_id}/analyze")
async def start_analysis(video_id: str, background_tasks: BackgroundTasks):
    if video_id not in analysis_status:
        # Check if it exists in storage
        vids = get_all_videos_list(STORAGE_DIR)
        v_match = next((v for v in vids if v["video_id"] == video_id), None)
        if v_match:
            analysis_status[video_id] = {
                "status": "uploaded",
                "progress": 0,
                "stages": ["Restored from database"],
                "filename": v_match["filename"],
                "camera_id": "CAM_01",
                "location": "Unknown"
            }
        else:
            raise HTTPException(status_code=404, detail="Video not found")
        
    analysis_status[video_id]["status"] = "processing"
    
    if DEMO_MODE:
        background_tasks.add_task(mock_analysis_task, video_id)
    else:
        background_tasks.add_task(run_real_analysis_task, video_id)
        
    return {"status": "started"}

async def mock_analysis_task(video_id: str):
    stages = [
        "✓ Video metadata extracted",
        "✓ Frames extracted",
        "○ YOLO object detection",
        "○ Person / vehicle tracking",
        "○ Attribute recognition",
        "○ Action recognition",
        "○ Generating embeddings",
        "○ Building FAISS index",
        "○ Database indexing"
    ]
    for i, stage in enumerate(stages):
        await asyncio.sleep(0.5)
        analysis_status[video_id]["stages"].append(stage)
        analysis_status[video_id]["progress"] = int(((i + 1) / len(stages)) * 100)
        
    analysis_status[video_id]["status"] = "completed"

def run_real_analysis_task(video_id: str):
    video_dir = os.path.join(STORAGE_DIR, "videos", video_id)
    files = os.listdir(video_dir) if os.path.exists(video_dir) else []
    if not files:
        analysis_status[video_id]["status"] = "error"
        return
        
    filepath = os.path.join(video_dir, files[0])
    
    try:
        analysis_status[video_id]["stages"].append("○ YOLO object detection")
        analysis_status[video_id]["progress"] = 20
        
        results = run_pipeline(
            source=filepath,
            model_registry=model_registry,
            conf=0.15,
            show=False
        )
        
        analysis_status[video_id]["stages"].append("○ Building FAISS index")
        analysis_status[video_id]["progress"] = 80
        
        if "detections" in results:
            before_vectors = vector_store.index.ntotal if vector_store and vector_store.index else 0
            persisted = _persist_detection_results(video_id, results)
            after_vectors = vector_store.index.ntotal if vector_store and vector_store.index else before_vectors
            print(f"Video {video_id} -> READY")
            print(f"FAISS vectors: {before_vectors} -> {after_vectors}")
            print(f"Persisted records: {len(persisted)}")
            
        analysis_status[video_id]["status"] = "completed"
        analysis_status[video_id]["progress"] = 100
        analysis_status[video_id]["stages"].append("✓ Ready")
        
    except Exception as e:
        print(f"Error analyzing {video_id}: {e}")
        analysis_status[video_id]["status"] = "error"


class ExtendedScanRequest(BaseModel):
    object: str
    concepts: Optional[List[str]] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    force: bool = False


def run_extended_analysis_task(video_id, query, scan_id):
    extended_scan_status[scan_id] = {
        "scan_id": scan_id,
        "video_id": video_id,
        "status": "processing",
        "progress": 5,
        "object": query.get("object"),
        "source_model": "yoloe",
        "stages": ["YOLOE scan queued"],
    }
    try:
        filepath = _video_path(video_id)
        if not filepath:
            raise FileNotFoundError(f"Video {video_id} not found")

        from config import YOLOE_MODEL_PATH
        from detectors.yoloe_detector import YOLOEDetector
        from extended_object_scan import run_extended_scan

        if "yoloe_detector" not in model_registry:
            extended_scan_status[scan_id]["stages"].append("Loading YOLOE lazily")
            model_registry["yoloe_detector"] = YOLOEDetector(YOLOE_MODEL_PATH)

        extended_scan_status[scan_id]["stages"].append("Running query-specific YOLOE detection")
        extended_scan_status[scan_id]["progress"] = 25
        results = run_extended_scan(
            source=filepath,
            video_id=video_id,
            query=query,
            detector=model_registry["yoloe_detector"],
            model_registry=model_registry,
        )
        extended_scan_status[scan_id]["progress"] = 80
        persisted = _persist_detection_results(video_id, results)
        extended_scan_status[scan_id].update(
            {
                "status": "completed",
                "progress": 100,
                "stages": ["YOLOE scan complete", "Tracks persisted"],
                "tracks": len(persisted),
                "elapsed_seconds": results.get("elapsed_seconds"),
            }
        )
    except Exception as exc:
        print(f"Error in YOLOE scan {scan_id}: {exc}")
        extended_scan_status[scan_id].update(
            {
                "status": "error",
                "progress": 100,
                "error": str(exc),
                "stages": ["YOLOE scan failed"],
            }
        )


def _queue_extended_scan(video_id, query, background_tasks, force=False):
    if not force and has_open_vocab_tracks(video_id, query.get("object") or "object"):
        return {
            "status": "completed",
            "reused": True,
            "video_id": video_id,
            "object": query.get("object"),
        }

    scan_id = f"scan_{uuid.uuid4().hex[:10]}"
    extended_scan_status[scan_id] = {
        "scan_id": scan_id,
        "video_id": video_id,
        "status": "queued",
        "progress": 0,
        "object": query.get("object"),
        "source_model": "yoloe",
        "stages": ["YOLOE scan queued"],
    }
    background_tasks.add_task(run_extended_analysis_task, video_id, query, scan_id)
    return {
        "status": "started",
        "scan_id": scan_id,
        "video_id": video_id,
        "object": query.get("object"),
        "concepts": query.get("concepts", []),
    }


@app.post("/videos/{video_id}/extended-scan")
async def start_extended_scan(video_id: str, req: ExtendedScanRequest, background_tasks: BackgroundTasks):
    available_videos = get_all_videos_list(STORAGE_DIR)
    if not any(video["video_id"] == video_id for video in available_videos):
        raise HTTPException(status_code=404, detail="Video not found")

    normalized = normalize_object_request(req.object)
    object_name = normalized.get("object") or req.object.lower().strip()
    concepts = req.concepts or normalized.get("concepts") or [object_name]
    query = {
        "object": object_name,
        "concepts": concepts,
        "time_start": req.time_start,
        "time_end": req.time_end,
    }

    response = _queue_extended_scan(video_id, query, background_tasks, force=req.force)
    if response.get("reused"):
        response["tracks"] = _find_existing_extended_tracks(video_id, [object_name])
    return response


@app.get("/extended-scans/{scan_id}")
async def get_extended_scan_status(scan_id: str):
    return extended_scan_status.get(
        scan_id,
        {"scan_id": scan_id, "status": "not_found", "progress": 0},
    )

@app.get("/videos/{video_id}/status")
async def get_status(video_id: str):
    if video_id not in analysis_status:
        return {"status": "failed", "progress": 0, "stages": ["Analysis failed — backend connection lost or restarted."]}
    return analysis_status[video_id]

@app.get("/videos/{video_id}/mapping-debug")
async def mapping_debug(video_id: str):
    """Diagnostic endpoint to check track/event data for a video without going through chat."""
    import sqlite3
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "crimevision.db"))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM events WHERE video_id = ?", (video_id,))
    event_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM tracks WHERE video_id = ?", (video_id,))
    track_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM track_points WHERE video_id = ?", (video_id,))
    tp_count = c.fetchone()[0]
    conn.close()
    return {
        "video_id": video_id,
        "events": event_count,
        "tracks": track_count,
        "track_points": tp_count
    }

@app.get("/videos/{video_id}/tracks")
async def get_tracks_endpoint(video_id: str):
    """
    Returns the complete dense tracking and event structure for a video.
    """
    available_videos = get_all_videos_list(STORAGE_DIR)
    video_meta = next((v for v in available_videos if v["video_id"] == video_id), None)
    if not video_meta:
        raise HTTPException(status_code=404, detail="Video not found")
        
    tracks = get_video_tracks(video_id)
    events = get_all_events(video_id)
    
    return {
        "video_id": video_id,
        "filename": video_meta["filename"],
        "duration": video_meta["duration"],
        "fps": video_meta.get("fps", 24.0),
        "frame_count": video_meta.get("frame_count", int(video_meta["duration"] * 24)),
        "tracks": tracks,
        "events": events
    }


class ChatMessage(BaseModel):
    role: str
    content: str
    results: Optional[List[dict]] = None
    
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    video_id: str
    extended_object_detection: bool = False
    session_id: Optional[str] = None

async def _chat_core(req: ChatRequest, background_tasks: BackgroundTasks):
    user_text = req.messages[-1].content.lower()
    
    # Contextual check: "show me the video"
    if "video" in user_text and "show" in user_text:
        if len(req.messages) > 1 and req.messages[-2].role == "assistant" and req.messages[-2].results:
            matches = req.messages[-2].results
            return JSONResponse(content={
                "response": "Here is the evidence you requested.",
                "matches": matches,
                "filters": {}
            })
            
    # Also handle "why" requests
    if user_text.startswith("show me why you identified this. [internal context:"):
        event_id = user_text.split("event_id=")[1].replace("]", "").strip()
        event = get_event_by_id(event_id)
        if event:
            c = event.get('confidence', 0) * 100
            why_text = f"Matched because:\n• A {event.get('object', 'object')} was detected.\n"
            
            attrs = event.get("attributes", {})
            if attrs.get("vehicle_color") and attrs.get("vehicle_color") != "Unknown":
                why_text += f"• Vehicle color classifier identified {attrs['vehicle_color']}.\n"
            if attrs.get("shirt_color") and attrs.get("shirt_color") != "Unknown":
                why_text += f"• Person attribute classifier identified {attrs['shirt_color']} shirt.\n"
                
            why_text += f"• Detection confidence: {int(c)}%.\n"
            return JSONResponse(content={
                "response": why_text,
                "matches": [],
                "filters": {}
            })

    # Context Resolution
    available_videos = get_all_videos_list(STORAGE_DIR)
    resolved_video_id = req.video_id
    
    # Priority A: Explicit filename in user's message
    has_explicit_filename = False
    for v in available_videos:
        if v["filename"].lower() in user_text:
            resolved_video_id = v["video_id"]
            has_explicit_filename = True
            break
            
    # Priority A.2: Cross-video explicit mention
    if any(k in user_text for k in ["across all", "any video", "all cameras", "all videos", "any camera"]):
        resolved_video_id = "ALL"
    elif not has_explicit_filename:
        # Smart Default: if the query is general and does not refer to "this video"/"here", default to ALL.
        local_keywords = ["this video", "this camera", "here", "selected video", "current video", "the video", "this clip"]
        if not any(k in user_text for k in local_keywords):
            resolved_video_id = "ALL"
        
    # Priority D: Previously referenced video in conversation (if req.video_id is empty)
    if not resolved_video_id:
        for msg in reversed(req.messages[:-1]):
            msg_text = msg.content.lower()
            for v in available_videos:
                if v["filename"].lower() in msg_text:
                    resolved_video_id = v["video_id"]
                    break
            if resolved_video_id: break

    # Priority E: Ambiguity Gate — only fires if NO video_id was provided by frontend dropdown
    if not resolved_video_id and len(available_videos) > 1:
        ambiguous_keywords = ["what happened", "analyze", "map", "show", "find", "track", "did"]
        if any(k in user_text for k in ambiguous_keywords):
            vid_list_str = "\n".join([f"• {v['filename']}" for v in available_videos])
            return JSONResponse(content={
                "response": f"I have multiple CCTV videos available. Which one would you like me to analyze?\n\n{vid_list_str}",
                "matches": [],
                "filters": {}
            })
    elif not resolved_video_id and len(available_videos) == 1:
        resolved_video_id = available_videos[0]["video_id"]
    
    # Final fallback: if still no resolved_video_id, log and bail
    if not resolved_video_id or resolved_video_id == "ALL":
        pass  # Allow downstream handling
    
    print(f"FULL MAPPING DEBUG — resolved_video_id={resolved_video_id}, req.video_id={req.video_id}")

    mapping_keywords = ["full analysis", "entire video", "what happened", "show me everything", "map this video", "analyze everything", "all detections", "all vehicles and people", "map", "fully mapped", "full map"]
    is_mapping_request = any(k in user_text for k in mapping_keywords)
    early_filters = optimize_prompt(user_text)
    normalized_request = normalize_object_request(user_text)
    requested_objects = extract_object_requests(user_text)
    explicit_extended_objects = [
        object_name for object_name in requested_objects
        if normalize_object_request(object_name).get("requires_open_vocab")
    ]
    # Accident/fight/etc. are persisted incident events, not open-vocabulary
    # objects. Only queue YOLOE when the request actually contains an explicit
    # open-vocabulary object alongside the incident query.
    if early_filters.get("event_type") and not normalized_request.get("object"):
        explicit_extended_objects = []
    if is_mapping_request and req.extended_object_detection and not explicit_extended_objects:
        explicit_extended_objects = list(ELECTRONIC_OBJECTS)

    # Explicit arbitrary-object requests, or the Extended Object Detection
    # control, queue YOLOE scans without rerunning YOLO11.
    if explicit_extended_objects:
        target_video_ids = (
            [v["video_id"] for v in available_videos]
            if resolved_video_id == "ALL"
            else ([resolved_video_id] if resolved_video_id else [])
        )
        queued_scans = []
        for target_video_id in target_video_ids:
            for object_name in explicit_extended_objects:
                object_request = normalize_object_request(object_name)
                query = {
                    "object": object_request.get("object") or object_name,
                    "concepts": object_request.get("concepts") or [object_name],
                    "time_start": early_filters.get("time_start"),
                    "time_end": early_filters.get("time_end"),
                }
                queued_scans.append(
                    _queue_extended_scan(target_video_id, query, background_tasks)
                )

        pending_scans = [scan for scan in queued_scans if scan.get("status") == "started"]
        if pending_scans:
            return JSONResponse(
                content={
                    "response": "Extended Object Detection is scanning the requested objects. I will keep the selected video context and expose the results when the scan completes.",
                    "matches": [],
                    "filters": {
                        "intent": "full_mapping" if is_mapping_request else "extended_scan",
                        "object": normalized_request.get("object"),
                        "objects": explicit_extended_objects,
                        "requires_open_vocab": True,
                    },
                    "extended_scan": {
                        "scan_ids": [scan["scan_id"] for scan in pending_scans],
                        "video_ids": target_video_ids,
                        "map_requested": is_mapping_request,
                        "query": user_text,
                    },
                }
            )

    if is_mapping_request:
        print(f"FULL MAPPING REQUEST — resolved_video_id={resolved_video_id}")
        
        tracks = []
        if resolved_video_id and resolved_video_id != "ALL":
            tracks = get_video_tracks(resolved_video_id)
            print(f"FULL MAPPING — tracks from get_video_tracks({resolved_video_id}): {len(tracks)}")
        
        # Always fetch events for this video
        events = get_all_events(resolved_video_id) if resolved_video_id and resolved_video_id != "ALL" else []
        print(f"FULL MAPPING — events for {resolved_video_id}: {len(events)}")
        
        # If no tracks AND no events, the video genuinely has no data
        if not tracks and not events:
            return JSONResponse(content={"response": "No track data found. Ensure the video has finished processing.", "matches": [], "filters": {}})
        
        # Build mapping from tracks if available, otherwise fall back to events
        if tracks:
            # ---- TRACK-BASED MAPPING (dense track_points available) ----
            persons = [t for t in tracks if t["class_name"].lower() == "person"]
            cars = [t for t in tracks if t["class_name"].lower() == "car"]
            motorcycles = [t for t in tracks if t["class_name"].lower() == "motorcycle"]
            other = [t for t in tracks if t["class_name"].lower() not in ("person", "car", "motorcycle")]
            
            duration = 0
            if tracks:
                last_seen_vals = [t.get("last_seen", 0) for t in tracks if t.get("last_seen")]
                if last_seen_vals:
                    duration = max(last_seen_vals)
                
            class_counts = {}
            for track in tracks:
                class_name = track.get("class_name", "unknown").capitalize()
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            md = f"### 🎬 FULL VIDEO MAP\n\n**Duration:** {duration:.1f} seconds\n\n"
            md += "**Objects Detected:**\n"
            md += "".join(f"• {count} {name}\n" for name, count in sorted(class_counts.items()))
            md += "\n**Tracks:**\n"
            for track in tracks[:50]:
                color = (track.get("color") or "Unknown").capitalize()
                brand = track.get("brand") or "Unknown"
                model = track.get("model") or "Unknown"
                speed = "Stationary" if track.get("stationary") else f"{track.get('average_speed', 0.0):.1f} px/s"
                source = track.get("source_model", "yolo11")
                first_s = track.get("first_seen", 0)
                last_s = track.get("last_seen", 0)
                md += (
                    f"• {track.get('class_name', 'Object').capitalize()} #{track.get('track_id')} "
                    f"— {color} — {brand} {model} — {speed} — {source} — "
                    f"visible {first_s:.1f}s to {last_s:.1f}s\n"
                )
            
            result_matches = tracks
        else:
            # ---- EVENTS-BASED FALLBACK (legacy videos without track_points) ----
            print(f"FULL MAPPING FALLBACK — using events for {resolved_video_id} ({len(events)} events)")
            
            # Group events by track_id to simulate tracks
            track_groups = {}
            for e in events:
                group_key = e.get("track_uid") or e.get("event_id", "unknown")
                track_groups.setdefault(group_key, []).append(e)
            
            persons = []
            cars = []
            motorcycles = []
            other_objs = []
            for group_key, evts in track_groups.items():
                obj_class = evts[0].get("object", "unknown").lower()
                attrs = evts[0].get("attributes", {})
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except:
                        attrs = {}
                entry = {
                    "track_id": evts[0].get("track_id", -1),
                    "track_uid": group_key,
                    "class_name": obj_class,
                    "color": attrs.get("vehicle_color") or attrs.get("shirt_color") or "Unknown",
                    "brand": attrs.get("vehicle_make", "Unknown"),
                    "model": attrs.get("vehicle_model", "Unknown"),
                    "source_model": evts[0].get("source_model", "yolo11"),
                    "stationary": bool(evts[0].get("stationary", False)),
                    "first_seen": min(e.get("timestamp", 0) for e in evts),
                    "last_seen": max(e.get("timestamp", 0) for e in evts),
                    "description": evts[0].get("description", "")
                }
                if obj_class == "person":
                    persons.append(entry)
                elif obj_class == "car":
                    cars.append(entry)
                elif obj_class == "motorcycle":
                    motorcycles.append(entry)
                else:
                    other_objs.append(entry)
            
            duration = max((e.get("timestamp", 0) for e in events), default=0)
            
            md = f"### 🎬 FULL VIDEO MAP\n\n**Duration:** {duration:.1f} seconds\n\n"
            md += f"**Objects Detected:**\n• {len(persons)} persons\n• {len(cars)} cars\n• {len(motorcycles)} motorcycles"
            if other_objs:
                md += f"\n• {len(other_objs)} other objects"
            md += "\n\n"
            
            md += "**Vehicles:**\n"
            for c in (cars + motorcycles)[:15]:
                color = c.get("color", "Unknown").capitalize()
                brand = c.get("brand", "Unknown")
                first_s = c.get("first_seen", 0)
                last_s = c.get("last_seen", 0)
                md += f"• {c['class_name'].capitalize()} #{c['track_id']} — {color} — {brand} — visible {first_s:.1f}s to {last_s:.1f}s\n"
                
            md += "\n**People:**\n"
            for p in persons[:15]:
                color = p.get("color", "Unknown").capitalize()
                first_s = p.get("first_seen", 0)
                last_s = p.get("last_seen", 0)
                md += f"• Person #{p['track_id']} — {color} shirt — visible {first_s:.1f}s to {last_s:.1f}s\n"
            
            # Build pseudo-track entries from events for result_matches
            result_matches = [e for e in events]
        
        # Incidents section (shared by both paths)
        incidents = [e for e in events if (e.get("event_type") or "").startswith("possible_")]
        md += "\n**Incidents:**\n"
        if incidents:
            for inc in incidents:
                ts = inc.get('timestamp', 0) or 0
                et = inc.get('event_type') or 'unknown'
                md += f"• {ts:.1f}s: {et.replace('_', ' ').capitalize()} detected\n"
        else:
            md += "• No confirmed accident\n• No confirmed fight\n"
            
        md += "\n\n[VIEW ANNOTATED VIDEO]\n[VIEW TIMELINE]\n[VIEW ALL TRACKS]"
        
        # Get filename
        filename = "unknown.mp4"
        for v in available_videos:
            if v["video_id"] == resolved_video_id:
                filename = v["filename"]
                break
        return JSONResponse(content={
            "response": md,
            "matches": result_matches,
            "filters": {
                "intent": "full_mapping",
                "full_mapping_data": {
                    "video_id": resolved_video_id,
                    "filename": filename,
                    "tracks": tracks if tracks else result_matches
                }
            }
        })

    context = {}
    if len(req.messages) > 1:
        prev_user_msgs = [m.content for m in req.messages if m.role == "user"]
        if len(prev_user_msgs) > 1:
            context = optimize_prompt(prev_user_msgs[-2])
            
    filters = optimize_prompt(user_text, context)
    q = filters.get("normalized_query", user_text)
    
    matches = _search_logic(q, resolved_video_id, filters)
    
    response_text = ""
    intent = filters.get("intent")
    
    if intent == "seek_map":
        filename = "unknown.mp4"
        for v in available_videos:
            if v["video_id"] == resolved_video_id:
                filename = v["filename"]
                break
        tracks = get_video_tracks(resolved_video_id) if resolved_video_id and resolved_video_id != "ALL" else []
        filters["full_mapping_data"] = {
            "video_id": resolved_video_id,
            "filename": filename,
            "tracks": tracks if tracks else matches
        }
        return JSONResponse(content={
            "response": "Opening the Video Map directly to your request...",
            "matches": matches,
            "filters": filters
        })
    
    if not matches:
        response_text = "No matching evidence was found across the indexed footage."
    else:
        m = matches[0]
        if intent == "explain":
            response_text = f"I identified this as a matching {filters.get('object', 'event')} because:\n"
            response_text += f"• A {m.get('object', 'object')} was detected in frame {m.get('frame')}.\n"
            if m.get("attributes", {}).get("shirt_color") or m.get("attributes", {}).get("vehicle_color"):
                response_text += f"• The extracted attributes matched the query.\n"
            if m.get("similarity"):
                response_text += f"• The visual embedding had a similarity of {m.get('similarity', 0):.2f}.\n"
            response_text += f"• The detection confidence was {int(m.get('confidence', 0) * 100)}%.\n"
            response_text += f"Evidence: Video {m.get('video_name', 'Unknown')} at {m.get('timestamp')}s."
        elif intent == "show_evidence":
            response_text = f"Here is the evidence from {m.get('video_name', 'Unknown')}."
        elif filters.get("sort") == "first":
            response_text = f"The first matching {filters.get('object', 'event')} appeared in {m.get('video_name', 'Unknown')} at {m.get('timestamp')} seconds."
        else:
            vids = len(set([mx.get("video_id") for mx in matches]))
            
            # Format custom response if brand/model was specifically searched
            if filters.get("brand"):
                response_text = f"I found {len(matches)} possible {filters['brand'].capitalize()} vehicles.\n\n"
                for i, match in enumerate(matches[:3]):
                    brand = match.get("attributes", {}).get("vehicle_make", "Unknown")
                    model = match.get("attributes", {}).get("vehicle_model", "Unknown")
                    b_conf = match.get("brand_confidence", 0)
                    
                    if b_conf < 0.75:
                        label = f"Possible {filters['brand'].capitalize()} — low confidence ({int(b_conf*100)}%)"
                    elif model and model != "Unknown":
                        label = f"{brand} {model} — {int(b_conf*100)}%"
                    else:
                        label = f"{brand} — model uncertain — {int(b_conf*100)}%"
                        
                    time_s = match.get("timestamp", 0)
                    response_text += f"{i+1}. {label} — {time_s:.1f}s\n"
            else:
                response_text = f"I found {len(matches)} matching observations across {vids} video(s)."
            
    return JSONResponse(content={
        "response": response_text,
        "matches": matches,
        "filters": filters
    })


@app.post("/chat")
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    session_id = req.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        
    user_msg = req.messages[-1]
    
    if not req.session_id:
        title = user_msg.content[:40] + ("..." if len(user_msg.content) > 40 else "")
        create_chat_session(session_id, title, req.video_id)
        
    insert_chat_message(session_id, "user", user_msg.content)
    
    response = await _chat_core(req, background_tasks)
    
    response_text = ""
    matches = []
    status_code = 200
    
    if isinstance(response, JSONResponse):
        body = json.loads(response.body.decode('utf-8'))
        status_code = response.status_code
    elif isinstance(response, dict):
        body = response
    else:
        return response
        
    response_text = body.get("response", "")
    matches = body.get("matches", [])
    
    insert_chat_message(session_id, "assistant", response_text, results=matches)
    
    body["session_id"] = session_id
    
    if isinstance(response, JSONResponse):
        return JSONResponse(content=body, status_code=status_code)
    return body


@app.get("/chat/sessions")
async def get_sessions(video_id: Optional[str] = None):
    try:
        sessions = get_chat_sessions(video_id)
        return JSONResponse(content=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    try:
        messages = get_chat_messages(session_id)
        return JSONResponse(content=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        delete_chat_session(session_id)
        return JSONResponse(content={"status": "success", "message": f"Session {session_id} deleted."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    query: str
    video_id: str
    
@app.post("/search")
async def search_evidence_post(req: SearchRequest):
    return await search_evidence(req.query, req.video_id)

@app.get("/search")
async def search_evidence(q: str, video_id: str):
    filters = optimize_prompt(q)
    norm_q = filters.get("normalized_query", q)
    matches = _search_logic(norm_q, video_id, filters)
    return JSONResponse(content={"query": q, "filters": filters, "matches": matches})

def _search_logic(q: str, video_id: str, filters: dict):
    if DEMO_MODE:
        matches = []
        demo_path = os.path.join(os.path.dirname(__file__), "demo_video_events.json")
        if os.path.exists(demo_path):
            with open(demo_path, "r") as f:
                events = json.load(f)
                
            for e in events:
                # Mock filter by video_id: in demo all events belong to demo_001
                if video_id and video_id != "demo_001" and e.get("video_id") != video_id:
                    pass # just for demo, we might want to return all anyway if testing
                    
                if filters.get("object") and e.get("object") != filters["object"]:
                    continue
                attrs = e.get("attributes", {})
                if filters.get("color"):
                    c = filters["color"]
                    if attrs.get("shirt_color") != c and attrs.get("vehicle_color") != c:
                        continue
                if filters.get("action") and e.get("action") != filters["action"]:
                    continue
                
                if filters.get("time_start") is not None and e["timestamp"] < filters["time_start"]:
                    continue
                if filters.get("time_end") is not None and e["timestamp"] > filters["time_end"]:
                    continue
                    
                matches.append(e)
                
        if filters.get("sort") == "first":
            matches = sorted(matches, key=lambda x: x["timestamp"])
            if matches:
                matches = [matches[0]]
        return matches
    else:
        query_emb = text_embedder.embed_text(q) if text_embedder else None
        from vector_store import FAISS_AVAILABLE
        use_faiss = query_emb is not None and FAISS_AVAILABLE and getattr(vector_store, "index", None) is not None
        
        if use_faiss:
            faiss_matches = vector_store.search(query_emb, k=50) # Larger pool for intersection
            if video_id != "ALL":
                faiss_matches = [m for m in faiss_matches if m.get("video_id") == video_id]
        else:
            faiss_matches = []
            
        filters["video_id"] = video_id
        db_matches = query_events(filters)
        
        # Hybrid Intersect
        if use_faiss:
            db_event_ids = {m["event_id"] for m in db_matches}
            matches = [m for m in faiss_matches if m["event_id"] in db_event_ids]
            
            # Semantic fallback logic for incident events (User Correction #3)
            if filters.get("event_type"):
                # Always prioritize the structured event matches if they exist
                if db_matches:
                    matches = db_matches
                else:
                    # If structured fails, rely on raw FAISS semantic matches
                    matches = faiss_matches
            else:
                # Fallback if no intersection and semantic isn't strictly required
                if not matches and not (filters.get("object") or filters.get("color")):
                    matches = db_matches
        else:
            matches = db_matches
            
        # Deduplicate to ensure we return unique objects across videos, rather than 50 frames of the same object
        unique_matches = {}
        for m in matches:
            # A numeric detector-local ID is not a global identity.  Prefer
            # the namespaced UID so YOLO11 and YOLOE results can never
            # collapse into one search result.
            k = m.get("track_uid")
            if not k:
                t_id = m.get("track_id")
                k = f"{m.get('video_id')}_{t_id}" if (t_id is not None and t_id != -1) else m.get("event_id")
            if k not in unique_matches or m.get("confidence", 0) > unique_matches[k].get("confidence", 0):
                unique_matches[k] = m
        
        matches = list(unique_matches.values())
        
        if filters.get("sort") == "first":
            matches = sorted(matches, key=lambda x: x["timestamp"])
            if matches:
                matches = [matches[0]]
                
        # Sort by combination of timestamp and confidence
        if not filters.get("sort"):
             matches = sorted(matches, key=lambda x: (x.get("incident_confidence") or x.get("confidence", 0)), reverse=True)
                
        return matches[:50]

@app.get("/generate-fir")
async def generate_fir(video_id: str = Query(...)):
    events = get_all_events(video_id)
    report = generate_fir_text(events, video_id)
    return JSONResponse(content={"report": report})

@app.get("/videos/{video_id}/events")
async def get_video_events(video_id: str):
    if DEMO_MODE:
        demo_path = os.path.join(os.path.dirname(__file__), "demo_video_events.json")
        if os.path.exists(demo_path):
            with open(demo_path, "r") as f:
                return JSONResponse(content=json.load(f))
    return JSONResponse(content=get_all_events(video_id))

@app.get("/videos/{video_id}/timeline")
async def get_video_timeline(video_id: str):
    if DEMO_MODE:
        demo_path = os.path.join(os.path.dirname(__file__), "demo_video_events.json")
        if os.path.exists(demo_path):
            with open(demo_path, "r") as f:
                return JSONResponse(content=json.load(f))
    return JSONResponse(content=get_all_events(video_id))

@app.get("/clips/{event_id}")
async def get_clip(event_id: str):
    event = None
    if DEMO_MODE:
        demo_path = os.path.join(os.path.dirname(__file__), "demo_video_events.json")
        if os.path.exists(demo_path):
            with open(demo_path, "r") as f:
                events = json.load(f)
                for e in events:
                    if e["event_id"] == event_id:
                        event = e
                        break
    else:
        event = get_event_by_id(event_id)
        
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
        
    video_id = event.get("video_id", "demo_001")
    video_name = event.get("video_name", "surveillance_demo.mp4")
    video_path = os.path.join(STORAGE_DIR, "videos", video_id, video_name)
    
    if not os.path.exists(video_path):
        # Fallback to local
        fallback = os.path.join(os.path.dirname(__file__), video_name)
        if os.path.exists(fallback):
            video_path = fallback
        else:
            raise HTTPException(status_code=404, detail="Original video file not found")
            
    clip_filename = f"clip_{event_id}.mp4"
    clip_path = os.path.join(STORAGE_DIR, "clips", video_id)
    os.makedirs(clip_path, exist_ok=True)
    full_clip_path = os.path.join(clip_path, clip_filename)
    
    result_path = generate_clip(video_path, event.get("clip_start", 0), event.get("clip_end", 5), full_clip_path)
    if not result_path:
        raise HTTPException(status_code=500, detail="Failed to generate clip")
        
    from fastapi.responses import FileResponse
    return FileResponse(result_path, media_type="video/mp4")

@app.get("/events/{event_id}/evidence")
def get_evidence(event_id: str):
    """
    Returns the evidence video clip and highlighted frame URLs for an event.
    """
    evidence = get_or_create_evidence(event_id, STORAGE_DIR)
    if not evidence:
        raise HTTPException(status_code=404, detail="Event or video not found")
    return evidence

@app.get("/incidents/timeline")
async def get_incident_timeline(video_id: str):
    """
    Returns a chronological timeline of meaningful events (incidents and first-seen objects).
    """
    events = get_all_events(video_id)
    timeline = []
    
    # Track which objects we've already announced to avoid spam
    announced_tracks = set()
    
    for e in sorted(events, key=lambda x: x["timestamp"]):
        is_incident = bool(e.get("event_type"))
        track = e.get("track_id", -1)
        track_key = e.get("track_uid") or f"{e.get('source_model', 'yolo11')}:{track}"
        
        # Always include incidents
        if is_incident:
            timeline.append({
                "type": "incident",
                "timestamp": e["timestamp"],
                "event_type": e["event_type"],
                "description": e["description"],
                "confidence": e.get("incident_confidence", e.get("confidence", 0)),
                "event_id": e["event_id"]
            })
        # Include the first time we see a person or vehicle
        elif track != -1 and track_key not in announced_tracks and not e.get("event_type"):
            announced_tracks.add(track_key)
            
            # Use formatted description if available, else basic
            desc = e.get("description", "")
            if not desc:
                attrs = e.get("attributes", {})
                brand = attrs.get("vehicle_make")
                color = attrs.get("vehicle_color") if e.get("object") != "person" else attrs.get("shirt_color")
                parts = []
                if color and color != "Unknown": parts.append(color)
                if brand and brand != "Unknown": parts.append(brand)
                parts.append(e.get("object"))
                desc = " ".join(parts).capitalize() + " detected"
                
            timeline.append({
                "type": "object_first_seen",
                "timestamp": e["timestamp"],
                "description": desc,
                "track_id": track,
                "track_uid": e.get("track_uid"),
                "source_model": e.get("source_model", "yolo11"),
                "event_id": e["event_id"]
            })
            
    return JSONResponse(content={"timeline": timeline})

@app.get("/incidents/summary")
async def get_incident_summary(video_id: str):
    """
    Returns a concise text summary of the incident timeline.
    """
    events = get_all_events(video_id)
    incidents = [e for e in events if e.get("event_type")]
    
    if not incidents:
        return JSONResponse(content={"summary": "No notable incidents detected in this video."})
        
    summary = f"INCIDENT SUMMARY\n{len(incidents)} notable events detected.\n\n"
    
    # Sort incidents chronologically
    for inc in sorted(incidents, key=lambda x: x["timestamp"]):
        time_str = f"{int(inc['timestamp'] // 60):02d}:{int(inc['timestamp'] % 60):02d}"
        summary += f"[{time_str}] {inc.get('description', 'Unknown event')}\n"
        
    return JSONResponse(content={"summary": summary})

@app.get("/journey/{video_id}/{track_id}")
async def get_object_journey(video_id: str, track_id: int):
    """
    Returns the journey and track history of a specific object within a video.
    """
    events = get_all_events(video_id)
    track_events = [e for e in events if e.get("track_id") == track_id]
    
    if not track_events:
        return JSONResponse(content={"error": "Track not found"}, status_code=404)
        
    track_events = sorted(track_events, key=lambda x: x["timestamp"])
    
    first_event = track_events[0]
    last_event = track_events[-1]
    
    journey = {
        "track_id": track_id,
        "video_id": video_id,
        "object": first_event.get("object"),
        "description": first_event.get("description"),
        "first_seen": first_event["timestamp"],
        "last_seen": last_event["timestamp"],
        "history": []
    }
    
    for e in track_events:
        journey["history"].append({
            "timestamp": e["timestamp"],
            "location": e.get("location", "Location metadata unavailable"),
            "speed": e.get("speed"),
            "event_id": e["event_id"]
        })
        
    return JSONResponse(content={"journey": journey})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
