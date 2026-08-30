import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "crimevision.db")


def _ensure_column(cursor, table, column, definition):
    existing = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            video_id TEXT NOT NULL,
            video_name TEXT,
            camera_id TEXT,
            location TEXT,
            frame INTEGER,
            timestamp REAL NOT NULL,
            track_id INTEGER,
            object TEXT NOT NULL,
            action TEXT,
            action_confidence REAL,
            confidence REAL,
            bbox TEXT,
            shirt_color TEXT,
            pant_color TEXT,
            vehicle_color TEXT,
            vehicle_type TEXT,
            vehicle_make TEXT,
            vehicle_model TEXT,
            speed REAL,
            speed_unit TEXT,
            clip_start REAL,
            clip_end REAL,
            description TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            filename TEXT,
            duration REAL,
            fps REAL,
            frame_count INTEGER,
            is_archived INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Try adding new columns if table existed before
    try:
        c.execute('ALTER TABLE events ADD COLUMN action_confidence REAL')
        c.execute('ALTER TABLE events ADD COLUMN speed REAL')
        c.execute('ALTER TABLE events ADD COLUMN speed_unit TEXT')
    except sqlite3.OperationalError:
        pass # Columns already exist
        
    try:
        c.execute('ALTER TABLE events ADD COLUMN brand_confidence REAL')
        c.execute('ALTER TABLE events ADD COLUMN model_confidence REAL')
        c.execute('ALTER TABLE events ADD COLUMN first_seen REAL')
        c.execute('ALTER TABLE events ADD COLUMN last_seen REAL')
    except sqlite3.OperationalError:
        pass
        
    try:
        c.execute('ALTER TABLE events ADD COLUMN event_type TEXT')
        c.execute('ALTER TABLE events ADD COLUMN sub_event_type TEXT')
        c.execute('ALTER TABLE events ADD COLUMN incident_confidence REAL')
        c.execute('ALTER TABLE events ADD COLUMN incident_reason TEXT')
        c.execute('ALTER TABLE events ADD COLUMN involved_track_ids TEXT')
    except sqlite3.OperationalError:
        pass
        
    c.execute('''
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            first_seen REAL,
            last_seen REAL,
            color TEXT,
            brand TEXT,
            attributes TEXT,
            average_speed REAL,
            UNIQUE(video_id, track_id)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS track_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            frame_number INTEGER,
            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL,
            confidence REAL,
            speed REAL
        )
    ''')

    # Additive migrations for unified YOLO11/YOLOE identities and evidence.
    for column, definition in [
        ("track_uid", "TEXT"),
        ("source_model", "TEXT DEFAULT 'yolo11'"),
        ("query_concepts", "TEXT DEFAULT '[]'"),
        ("confidence", "REAL"),
        ("bbox", "TEXT"),
        ("model", "TEXT DEFAULT 'Unknown'"),
        ("max_speed", "REAL"),
        ("stationary", "INTEGER DEFAULT 0"),
        ("speed_status", "TEXT"),
    ]:
        _ensure_column(c, "tracks", column, definition)

    for column, definition in [
        ("track_uid", "TEXT"),
        ("source_model", "TEXT DEFAULT 'yolo11'"),
    ]:
        _ensure_column(c, "track_points", column, definition)

    for column, definition in [
        ("track_uid", "TEXT"),
        ("source_model", "TEXT DEFAULT 'yolo11'"),
        ("query_concepts", "TEXT DEFAULT '[]'"),
        ("color", "TEXT"),
        ("model", "TEXT DEFAULT 'Unknown'"),
        ("stationary", "INTEGER DEFAULT 0"),
    ]:
        _ensure_column(c, "events", column, definition)

    # Existing rows are YOLO11 rows. Preserve their data and only backfill
    # identity/source metadata required by the unified representation.
    c.execute("UPDATE tracks SET source_model = COALESCE(NULLIF(source_model, ''), 'yolo11')")
    c.execute("UPDATE tracks SET query_concepts = COALESCE(query_concepts, '[]')")
    c.execute(
        "UPDATE tracks SET track_uid = source_model || ':' || video_id || ':' || track_id "
        "WHERE track_uid IS NULL OR track_uid = ''"
    )
    c.execute("UPDATE track_points SET source_model = COALESCE(NULLIF(source_model, ''), 'yolo11')")
    c.execute(
        "UPDATE track_points SET track_uid = source_model || ':' || video_id || ':' || track_id "
        "WHERE track_uid IS NULL OR track_uid = ''"
    )
    c.execute("UPDATE events SET source_model = COALESCE(NULLIF(source_model, ''), 'yolo11')")
    c.execute("UPDATE events SET query_concepts = COALESCE(query_concepts, '[]')")
    c.execute(
        "UPDATE events SET track_uid = source_model || ':' || video_id || ':' || track_id "
        "WHERE track_uid IS NULL AND track_id IS NOT NULL AND track_id != -1"
    )
    c.execute("UPDATE events SET color = COALESCE(color, vehicle_color, shirt_color, 'Unknown')")

    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            video_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            results TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()

def insert_events(events_list):
    """
    Inserts a list of structured event dictionaries into the DB.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for evt in events_list:
        attrs = evt.get("attributes", {})
        source_model = evt.get("source_model") or "yolo11"
        track_id = evt.get("track_id")
        track_uid = evt.get("track_uid")
        if not track_uid and track_id is not None and track_id != -1:
            track_uid = f"{source_model}:{evt.get('video_id')}:{track_id}"
        color = (
            evt.get("color")
            or attrs.get("object_color")
            or attrs.get("vehicle_color")
            or attrs.get("shirt_color")
            or "Unknown"
        )
        c.execute('''
            INSERT OR IGNORE INTO events (
                event_id, video_id, video_name, camera_id, location, frame, timestamp, 
                track_id, object, action, action_confidence, confidence, bbox, 
                shirt_color, pant_color, vehicle_color, vehicle_type, vehicle_make, vehicle_model,
                speed, speed_unit, clip_start, clip_end, description,
                brand_confidence, model_confidence, first_seen, last_seen,
                event_type, sub_event_type, incident_confidence, incident_reason, involved_track_ids,
                track_uid, source_model, query_concepts, color, model, stationary
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            evt.get("event_id"),
            evt.get("video_id"),
            evt.get("video_name"),
            evt.get("camera_id"),
            evt.get("location"),
            evt.get("frame"),
            evt.get("timestamp"),
            evt.get("track_id"),
            evt.get("object"),
            evt.get("action"),
            evt.get("action_confidence"),
            evt.get("confidence"),
            json.dumps(evt.get("bbox", [])),
            attrs.get("shirt_color"),
            attrs.get("pants_color"),
            attrs.get("vehicle_color"),
            attrs.get("vehicle_type"),
            attrs.get("vehicle_make") or evt.get("brand"),
            attrs.get("vehicle_model") or evt.get("model"),
            evt.get("speed"),
            evt.get("speed_unit"),
            evt.get("clip_start"),
            evt.get("clip_end"),
            evt.get("description"),
            evt.get("brand_confidence"),
            evt.get("model_confidence"),
            evt.get("first_seen"),
            evt.get("last_seen"),
            evt.get("event_type"),
            evt.get("sub_event_type"),
            evt.get("incident_confidence"),
            json.dumps(evt.get("reason", [])) if evt.get("reason") else None,
            json.dumps(evt.get("involved_track_ids", [])) if evt.get("involved_track_ids") else None,
            track_uid,
            source_model,
            json.dumps(evt.get("query_concepts", [])),
            color,
            evt.get("model") or attrs.get("model") or "Unknown",
            1 if evt.get("stationary") else 0,
        ))
    
    conn.commit()
    conn.close()

def query_events(query_filters):
    """
    Queries events based on structured filters.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    base_query = "SELECT * FROM events WHERE 1=1"
    params = []
    
    # Apply filters dynamically
    video_id = query_filters.get("video_id")
    if video_id and video_id.lower() != "all":
        base_query += " AND video_id = ?"
        params.append(video_id)

    event_type = query_filters.get("event_type")
    if event_type:
        base_query += " AND event_type = ?"
        params.append(event_type)

    obj_type = query_filters.get("object")
    if obj_type:
        obj_type = obj_type.lower()
        if obj_type == "vehicle":
            base_query += " AND (LOWER(object) = 'car' OR LOWER(object) = 'truck' OR LOWER(object) = 'bus' OR LOWER(object) = 'motorcycle' OR LOWER(object) = 'suv' OR LOWER(object) = 'sedan')"
        elif obj_type == "person":
            base_query += " AND LOWER(object) = 'person'"
        else:
            base_query += " AND LOWER(object) LIKE ?"
            params.append(f"%{obj_type}%")
        
    color = query_filters.get("color")
    if color:
        color = color.lower()
        base_query += " AND (LOWER(color) LIKE ? OR LOWER(shirt_color) LIKE ? OR LOWER(pant_color) LIKE ? OR LOWER(vehicle_color) LIKE ?)"
        params.extend([f"%{color}%", f"%{color}%", f"%{color}%", f"%{color}%"])

    brand = query_filters.get("brand")
    if brand:
        brand = brand.lower()
        base_query += " AND LOWER(vehicle_make) LIKE ?"
        params.append(f"%{brand}%")

    if "weapon" in query_filters:
        base_query += " AND (LOWER(object) = 'knife' OR LOWER(object) = 'gun')"
        
    # Time window filters
    time_start = query_filters.get("time_start")
    if time_start is not None:
        base_query += " AND timestamp >= ?"
        params.append(time_start)
        
    time_end = query_filters.get("time_end")
    if time_end is not None:
        base_query += " AND timestamp <= ?"
        params.append(time_end)

    base_query += " ORDER BY timestamp ASC"
    
    c.execute(base_query, params)
    rows = c.fetchall()
    
    results = []
    for r in rows:
        # Check if columns exist (for backward compatibility if missing)
        def get_col(row, col_name, default=None):
            return row[col_name] if col_name in row.keys() else default

        results.append({
            "event_id": r["event_id"],
            "video_id": r["video_id"],
            "video_name": r["video_name"],
            "camera_id": r["camera_id"],
            "location": r["location"],
            "frame": r["frame"],
            "timestamp": r["timestamp"],
            "track_id": r["track_id"],
            "track_uid": get_col(r, "track_uid"),
            "object": r["object"],
            "action": r["action"],
            "action_confidence": r["action_confidence"],
            "confidence": r["confidence"],
            "bbox": json.loads(r["bbox"]) if r["bbox"] else [],
            "attributes": {
                "shirt_color": r["shirt_color"],
                "pants_color": r["pant_color"],
                "vehicle_color": r["vehicle_color"],
                "vehicle_type": r["vehicle_type"],
                "vehicle_make": r["vehicle_make"],
                "vehicle_model": r["vehicle_model"],
                "object_color": get_col(r, "color", "Unknown") or "Unknown",
                "model": get_col(r, "model", "Unknown") or "Unknown",
            },
            "speed": r["speed"],
            "speed_unit": r["speed_unit"],
            "clip_start": r["clip_start"],
            "clip_end": r["clip_end"],
            "description": r["description"],
            "brand_confidence": get_col(r, "brand_confidence"),
            "model_confidence": get_col(r, "model_confidence"),
            "first_seen": get_col(r, "first_seen"),
            "last_seen": get_col(r, "last_seen"),
            "event_type": get_col(r, "event_type"),
            "sub_event_type": get_col(r, "sub_event_type"),
            "incident_confidence": get_col(r, "incident_confidence"),
            "reason": json.loads(get_col(r, "incident_reason", "[]")) if get_col(r, "incident_reason") else [],
            "involved_track_ids": json.loads(get_col(r, "involved_track_ids", "[]")) if get_col(r, "involved_track_ids") else []
            ,"source_model": get_col(r, "source_model", "yolo11") or "yolo11"
            ,"query_concepts": json.loads(get_col(r, "query_concepts", "[]")) if get_col(r, "query_concepts") else []
            ,"color": get_col(r, "color", "Unknown") or "Unknown"
            ,"model": get_col(r, "model", "Unknown") or "Unknown"
            ,"stationary": bool(get_col(r, "stationary", 0))
        })
        
    conn.close()
    return results

def get_event_by_id(event_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
    r = c.fetchone()
    conn.close()
    
    if r:
        def get_col(row, col_name, default=None):
            return row[col_name] if col_name in row.keys() else default
            
        return {
            "event_id": r["event_id"],
            "video_id": r["video_id"],
            "video_name": r["video_name"],
            "camera_id": r["camera_id"],
            "location": r["location"],
            "frame": r["frame"],
            "timestamp": r["timestamp"],
            "track_id": r["track_id"],
            "track_uid": get_col(r, "track_uid"),
            "object": r["object"],
            "action": r["action"],
            "action_confidence": r["action_confidence"],
            "confidence": r["confidence"],
            "bbox": json.loads(r["bbox"]) if r["bbox"] else [],
            "attributes": {
                "shirt_color": r["shirt_color"],
                "pants_color": r["pant_color"],
                "vehicle_color": r["vehicle_color"],
                "vehicle_type": r["vehicle_type"],
                "vehicle_make": r["vehicle_make"],
                "vehicle_model": r["vehicle_model"],
                "object_color": get_col(r, "color", "Unknown") or "Unknown",
                "model": get_col(r, "model", "Unknown") or "Unknown",
            },
            "speed": r["speed"],
            "speed_unit": r["speed_unit"],
            "clip_start": r["clip_start"],
            "clip_end": r["clip_end"],
            "description": r["description"],
            "brand_confidence": get_col(r, "brand_confidence"),
            "model_confidence": get_col(r, "model_confidence"),
            "first_seen": get_col(r, "first_seen"),
            "last_seen": get_col(r, "last_seen"),
            "event_type": get_col(r, "event_type"),
            "sub_event_type": get_col(r, "sub_event_type"),
            "incident_confidence": get_col(r, "incident_confidence"),
            "reason": json.loads(get_col(r, "incident_reason", "[]")) if get_col(r, "incident_reason") else [],
            "involved_track_ids": json.loads(get_col(r, "involved_track_ids", "[]")) if get_col(r, "involved_track_ids") else []
            ,"source_model": get_col(r, "source_model", "yolo11") or "yolo11"
            ,"query_concepts": json.loads(get_col(r, "query_concepts", "[]")) if get_col(r, "query_concepts") else []
            ,"color": get_col(r, "color", "Unknown") or "Unknown"
            ,"model": get_col(r, "model", "Unknown") or "Unknown"
            ,"stationary": bool(get_col(r, "stationary", 0))
        }
    return None

def get_all_events(video_id):
    return query_events({"video_id": video_id})

def get_all_videos_list(storage_dir):
    """
    Returns every video present in the database or on disk, attempting to fetch real
    duration from metadata. The storage scan is important for uploaded videos that
    have not produced any detections yet (and therefore have no events to migrate).
    """
    import cv2
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
    storage_videos_dir = os.path.join(storage_dir, "videos")
    storage_files = {}
    if os.path.isdir(storage_videos_dir):
        for video_id in os.listdir(storage_videos_dir):
            video_dir = os.path.join(storage_videos_dir, video_id)
            if not os.path.isdir(video_dir):
                continue
            video_files = sorted(
                filename for filename in os.listdir(video_dir)
                if filename.lower().endswith(video_extensions)
            )
            if video_files:
                storage_files[video_id] = os.path.join(video_dir, video_files[0])

    def metadata_for(video_path, fallback_duration=0):
        duration = fallback_duration or 0
        fps_val = 24.0
        frame_count_val = int(duration * fps_val)
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0 and frame_count > 0:
                    duration = frame_count / fps
                    fps_val = fps
                    frame_count_val = int(frame_count)
            cap.release()
        return duration, fps_val, frame_count_val

    def stored_path(video_id, filename=None):
        preferred = os.path.join(storage_videos_dir, video_id, filename) if filename else None
        if preferred and os.path.exists(preferred):
            return preferred
        return storage_files.get(video_id)
    
    # 1. Ensure all videos from events are in the videos table (auto-migration)
    c.execute('''
        SELECT e.video_id, e.video_name, MAX(e.timestamp) as max_time 
        FROM events e
        LEFT JOIN videos v ON e.video_id = v.video_id
        WHERE v.video_id IS NULL
        GROUP BY e.video_id, e.video_name
    ''')
    missing_videos = c.fetchall()
    
    for v in missing_videos:
        video_id = v["video_id"]
        if not video_id:
            continue
            
        video_name = v["video_name"]
        if not video_name or video_name == "unknown.mp4":
            video_path = stored_path(video_id)
            if video_path:
                video_name = os.path.basename(video_path)
                c.execute("UPDATE events SET video_name = ? WHERE video_id = ?", (video_name, video_id))
        
        if not video_name:
            video_name = "unknown.mp4"
            
        vid_path = stored_path(video_id, video_name)
        duration, fps_val, frame_count_val = metadata_for(vid_path, v["max_time"])
            
        c.execute('''
            INSERT INTO videos (video_id, filename, duration, fps, frame_count, is_archived)
            VALUES (?, ?, ?, ?, ?, 0)
        ''', (video_id, video_name, duration, fps_val, frame_count_val))

    # 1b. Reconcile uploaded files that have no events/tracks yet. Without this,
    # a valid stored video disappears from the gallery until detection finds an event.
    c.execute("SELECT video_id, filename, duration, fps, frame_count FROM videos")
    known_videos = {row["video_id"]: row for row in c.fetchall()}
    for video_id, video_path in storage_files.items():
        filename = os.path.basename(video_path)
        existing = known_videos.get(video_id)
        if existing is None:
            duration, fps_val, frame_count_val = metadata_for(video_path)
            c.execute('''
                INSERT INTO videos (video_id, filename, duration, fps, frame_count, is_archived)
                VALUES (?, ?, ?, ?, ?, 0)
            ''', (video_id, filename, duration, fps_val, frame_count_val))
        elif not existing["filename"] or existing["filename"] == "unknown.mp4" or not os.path.exists(stored_path(video_id, existing["filename"])):
            duration, fps_val, frame_count_val = metadata_for(video_path, existing["duration"])
            c.execute('''
                UPDATE videos
                SET filename = ?, duration = ?, fps = ?, frame_count = ?
                WHERE video_id = ?
            ''', (filename, duration, fps_val, frame_count_val, video_id))
    
    conn.commit()

    # 2. Fetch all videos with aggregated stats
    # Use subqueries to count accurately without Cartesian products
    c.execute('''
        SELECT 
            v.video_id, v.filename, v.duration, v.fps, v.frame_count, v.is_archived,
            (SELECT COUNT(*) FROM tracks WHERE video_id = v.video_id) as tracks_count,
            (SELECT COUNT(*) FROM events WHERE video_id = v.video_id) as events_count,
            (SELECT COUNT(DISTINCT class_name) FROM tracks WHERE video_id = v.video_id) as unique_objects_count,
            (SELECT COUNT(*) FROM events WHERE video_id = v.video_id AND event_type IN ('possible_accident', 'confirmed_accident', 'fight', 'weapon_detected', 'incident')) as incidents_count
        FROM videos v
        ORDER BY v.created_at DESC
    ''')
    videos = c.fetchall()
    conn.close()
    
    result = []
    for v in videos:
        result.append({
            "video_id": v["video_id"],
            "filename": v["filename"],
            "duration": round(v["duration"], 1),
            "status": "READY",
            "fps": round(v["fps"], 1),
            "frame_count": v["frame_count"],
            "is_archived": bool(v["is_archived"]),
            "stats": {
                "tracks": v["tracks_count"],
                "events": v["events_count"],
                "objects": v["unique_objects_count"],
                "incidents": v["incidents_count"]
            }
        })
    return result

def set_video_archived(video_id, is_archived):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE videos SET is_archived = ? WHERE video_id = ?", (1 if is_archived else 0, video_id))
    conn.commit()
    conn.close()

def delete_video_data(video_id):
    """
    Deletes all records associated with a video_id from all tables.
    Returns a list of event_ids that were deleted so evidence files can be removed.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get all event_ids first
    c.execute("SELECT event_id FROM events WHERE video_id = ?", (video_id,))
    event_ids = [row[0] for row in c.fetchall()]
    
    # Delete from all tables
    c.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
    c.execute("DELETE FROM track_points WHERE video_id = ?", (video_id,))
    c.execute("DELETE FROM tracks WHERE video_id = ?", (video_id,))
    c.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
    
    conn.commit()
    conn.close()
    
    return event_ids


def get_next_track_id(video_id):
    """Allocate a video-local friendly ID without colliding across detector branches."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COALESCE(MAX(track_id), 0) + 1 FROM tracks WHERE video_id = ?", (video_id,))
    next_id = int(c.fetchone()[0])
    conn.close()
    return next_id


def has_open_vocab_tracks(video_id, object_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM tracks WHERE video_id = ? AND source_model = 'yoloe' AND LOWER(class_name) = LOWER(?)",
        (video_id, object_name),
    )
    result = c.fetchone()[0] > 0
    conn.close()
    return result

def insert_tracks(video_id, tracks_list):
    """
    Inserts raw tracking data into tracks and track_points tables.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for t in tracks_list:
        source_model = t.get("source_model") or "yolo11"
        track_id = int(t["track_id"])
        track_uid = t.get("track_uid") or f"{source_model}:{video_id}:{track_id}"
        attributes = t.get("attributes", {}) or {}
        color = (
            t.get("color")
            or attributes.get("object_color")
            or attributes.get("vehicle_color")
            or attributes.get("shirt_color")
            or "Unknown"
        )
        # Insert into tracks
        c.execute('''
            INSERT OR IGNORE INTO tracks (
                video_id, track_id, class_name, first_seen, last_seen,
                color, brand, attributes, average_speed, track_uid, source_model,
                query_concepts, confidence, bbox, model, max_speed, stationary, speed_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id,
            track_id,
            t["object"],
            t.get("first_seen"),
            t.get("last_seen"),
            color,
            t.get("brand") or attributes.get("vehicle_make", "Unknown"),
            json.dumps(attributes),
            t.get("average_speed", 0.0),
            track_uid,
            source_model,
            json.dumps(t.get("query_concepts", [])),
            t.get("confidence"),
            json.dumps(t.get("bbox", [])),
            t.get("model") or attributes.get("model", "Unknown"),
            t.get("max_speed", t.get("average_speed", 0.0)),
            1 if t.get("stationary") else 0,
            t.get("speed_unit") or ("Stationary" if t.get("stationary") else "pixels/second"),
        ))
        
        # Insert points
        points_data = []
        for p in t.get("points", []):
            points_data.append((
                video_id,
                track_id,
                p.get("timestamp"),
                p.get("frame"),
                p.get("bbox", [0,0,0,0])[0],
                p.get("bbox", [0,0,0,0])[1],
                p.get("bbox", [0,0,0,0])[2],
                p.get("bbox", [0,0,0,0])[3],
                p.get("confidence"),
                p.get("speed"),
                track_uid,
                p.get("source_model") or source_model,
            ))
            
        c.executemany('''
            INSERT INTO track_points (
                video_id, track_id, timestamp, frame_number, x1, y1, x2, y2, confidence, speed,
                track_uid, source_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', points_data)
        
    conn.commit()
    conn.close()

def get_video_tracks(video_id):
    """
    Retrieves all tracks and their points for a video.
    Returns a list of track dicts with a 'points' array.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM tracks WHERE video_id = ?", (video_id,))
    tracks = [dict(row) for row in c.fetchall()]
    
    c.execute("SELECT * FROM track_points WHERE video_id = ? ORDER BY timestamp ASC", (video_id,))
    points = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    # Associate points by unique identity, never by a detector-local numeric ID.
    tracks_map = {}
    for t in tracks:
        t["track_uid"] = t.get("track_uid") or f"{t.get('source_model', 'yolo11')}:{video_id}:{t['track_id']}"
        tracks_map[t["track_uid"]] = t
    for t in tracks_map.values():
        t["points"] = []
        t["attributes"] = json.loads(t["attributes"]) if t.get("attributes") else {}
        t["query_concepts"] = json.loads(t["query_concepts"]) if t.get("query_concepts") else []
        t["stationary"] = bool(t.get("stationary", 0))
        t["bbox"] = json.loads(t["bbox"]) if t.get("bbox") else []
        
    for p in points:
        point_uid = p.get("track_uid") or f"{p.get('source_model', 'yolo11')}:{video_id}:{p['track_id']}"
        if point_uid in tracks_map:
            p["bbox"] = [p["x1"], p["y1"], p["x2"], p["y2"]]
            tracks_map[point_uid]["points"].append(p)
            
    return list(tracks_map.values())


def create_chat_session(session_id, title, video_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO chat_sessions (session_id, title, video_id)
        VALUES (?, ?, ?)
    ''', (session_id, title, video_id))
    conn.commit()
    conn.close()

def insert_chat_message(session_id, role, content, results=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_messages (session_id, role, content, results)
        VALUES (?, ?, ?, ?)
    ''', (session_id, role, content, json.dumps(results) if results is not None else None))
    conn.commit()
    conn.close()

def get_chat_sessions(video_id=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if video_id and video_id.lower() != "all":
        c.execute('''
            SELECT * FROM chat_sessions 
            WHERE video_id = ? OR video_id = 'ALL'
            ORDER BY created_at DESC
        ''', (video_id,))
    else:
        c.execute('SELECT * FROM chat_sessions ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_chat_messages(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC', (session_id,))
    rows = c.fetchall()
    conn.close()
    
    results_list = []
    for r in rows:
        results_list.append({
            "role": r["role"],
            "content": r["content"],
            "results": json.loads(r["results"]) if r["results"] else None,
            "timestamp": r["timestamp"]
        })
    return results_list

def delete_chat_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM chat_messages WHERE session_id = ?', (session_id,))
    c.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()
