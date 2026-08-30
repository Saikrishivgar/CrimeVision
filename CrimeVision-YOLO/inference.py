import os
import cv2
import uuid
import time
import json
import torch
import numpy as np
import sys

from detector import YOLODetector
from video_processor import VideoProcessor
from attribute import AttributeExtractor
from embedder import Embedder
from vehicle_classifier import VehicleClassifier
from utils import calculate_distance, calculate_iou, calculate_center

def get_mode(lst):
    if not lst:
        return "Unknown"
    return max(set(lst), key=lst.count)

def detect_color_opencv(crop):
    """
    Fast OpenCV HSV color detection.
    """
    if crop is None or crop.size == 0:
        return "Unknown"
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Define color boundaries in HSV
    colors = {
        "Red": [
            (np.array([0, 70, 50]), np.array([10, 255, 255])),
            (np.array([170, 70, 50]), np.array([180, 255, 255]))
        ],
        "Orange": [(np.array([11, 70, 50]), np.array([25, 255, 255]))],
        "Yellow": [(np.array([26, 70, 50]), np.array([34, 255, 255]))],
        "Green": [(np.array([35, 70, 50]), np.array([85, 255, 255]))],
        "Blue": [(np.array([100, 70, 50]), np.array([130, 255, 255]))],
        "Purple": [(np.array([131, 70, 50]), np.array([169, 255, 255]))],
        "Black": [(np.array([0, 0, 0]), np.array([180, 255, 40]))],
        "White": [(np.array([0, 0, 200]), np.array([180, 40, 255]))],
        "Gray": [(np.array([0, 0, 40]), np.array([180, 40, 200]))]
    }
    
    best_color = "Unknown"
    max_pixels = 0
    total_pixels = max(crop.shape[0] * crop.shape[1], 1)
    
    for color_name, bounds in colors.items():
        color_pixels = 0
        for lower, upper in bounds:
            mask = cv2.inRange(hsv, lower, upper)
            color_pixels += cv2.countNonZero(mask)
            
        ratio = color_pixels / total_pixels
        if ratio > 0.12 and color_pixels > max_pixels:
            max_pixels = color_pixels
            best_color = color_name
            
    return best_color

def run_pipeline(source, model_registry=None, conf=0.15, show=False):
    """
    Main processing pipeline. Redesigned into TWO PASSES for extreme speed.
    """
    start_time = time.time()
    print("==================================================")
    print("PASS 1 — FAST VIDEO ANALYSIS")
    print("==================================================")
    
    try:
        processor = VideoProcessor(source)
    except ValueError as e:
        print(f"Error opening video: {e}")
        return {}
        
    print(f"Video duration: {processor.duration:.2f}s")
    print(f"FPS: {processor.fps:.0f}")
    
    total_frames = processor.total_frames
    # Sample every 10 frames (~3 FPS if 30 FPS video)
    sampled_frames_expected = total_frames // 10 if total_frames > 0 else 0
    print(f"Sampled frames: {sampled_frames_expected}")

    if model_registry and "detector" in model_registry:
        detector = model_registry["detector"]
    else:
        # Fallback for standalone script
        detector = YOLODetector("yolo11n.pt")
        
    print(f"YOLO device: {detector.model.device}")

    # Pass 1 State
    track_history = {}
    last_seen_untracked = {}
    synthetic_id_counter = 10000
    UNTRACKED_MERGE_THRESHOLD = 2.0 
    MAX_CROPS = 5

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
    PERSON_CLASSES = {"person"}

    sampled_frames_count = 0
    yolo_inference_time = 0
    color_analysis_time = 0

    print("Processing video frames...")
    
    try:
        for frame_idx, timestamp, frame in processor.read_frames():
            if frame_idx % 10 != 0:
                continue
                
            sampled_frames_count += 1
            current_seconds = frame_idx / processor.fps
            
            t0 = time.time()
            detections = detector.detect_and_track(frame, persist=True, conf=conf)
            yolo_inference_time += (time.time() - t0)
            
            for det in detections:
                class_name = det["class"]
                track_id = det["track_id"]
                bbox = det["bbox"]
                confidence = det["confidence"]
                
                if track_id is None:
                    recent = last_seen_untracked.get(class_name)
                    if recent and (current_seconds - recent["seconds"]) <= UNTRACKED_MERGE_THRESHOLD:
                        track_id = recent["id"]
                    else:
                        synthetic_id_counter += 1
                        track_id = synthetic_id_counter
                    
                    last_seen_untracked[class_name] = {
                        "timestamp": timestamp,
                        "seconds": current_seconds,
                        "id": track_id
                    }
                
                if track_id not in track_history:
                    track_history[track_id] = {
                        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
                        "track_id": track_id,
                        "object": class_name,
                        "first_timestamp": timestamp,
                        "first_seconds": current_seconds,
                        "last_seconds": current_seconds,
                        "confidences": [],
                        "centers": [],
                        "smoothed_bbox": list(bbox),
                        "crops_buffer": [], # list of (confidence, crop, type)
                        "opencv_colors": [],
                        "points": []
                    }
                else:
                    # EMA BBox Smoothing
                    alpha = 0.5
                    sb = track_history[track_id]["smoothed_bbox"]
                    track_history[track_id]["smoothed_bbox"] = [
                        alpha * bbox[0] + (1 - alpha) * sb[0],
                        alpha * bbox[1] + (1 - alpha) * sb[1],
                        alpha * bbox[2] + (1 - alpha) * sb[2],
                        alpha * bbox[3] + (1 - alpha) * sb[3]
                    ]
                
                track_history[track_id]["confidences"].append(confidence)
                track_history[track_id]["last_seconds"] = current_seconds
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                centers_list = track_history[track_id]["centers"]
                centers_list.append((cx, cy, current_seconds))
                
                # Calculate instantaneous speed (px/sec)
                inst_speed = 0.0
                if len(centers_list) > 1:
                    prev_cx, prev_cy, prev_t = centers_list[-2]
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    dt = current_seconds - prev_t
                    if dt > 0:
                        inst_speed = (dx**2 + dy**2)**0.5 / dt
                
                track_history[track_id]["points"].append({
                    "timestamp": current_seconds,
                    "frame": frame_idx,
                    "bbox": list(bbox),
                    "confidence": float(confidence),
                    "speed": round(inst_speed, 2)
                })
                
                # Intelligent Cropping
                x1, y1, x2, y2 = map(int, bbox)
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop_full = frame[y1:y2, x1:x2]
                
                intelligent_crop = crop_full
                crop_type = "full"
                
                if crop_full.size > 0:
                    ch, cw, _ = crop_full.shape
                    if class_name.lower() in PERSON_CLASSES:
                        # Upper body / shirt (avoid head and legs)
                        intelligent_crop = crop_full[int(0.15*ch):int(0.6*ch), int(0.1*cw):int(0.9*cw)]
                        crop_type = "shirt"
                    elif class_name.lower() in VEHICLE_CLASSES:
                        # Vehicle body (avoid roof, sky, wheels)
                        intelligent_crop = crop_full[int(0.3*ch):int(0.85*ch), int(0.1*cw):int(0.9*cw)]
                        crop_type = "vehicle"
                        
                    if intelligent_crop.size > 0:
                        buf = track_history[track_id]["crops_buffer"]
                        buf.append((confidence, intelligent_crop, crop_full))
                        # Keep only top N crops by confidence
                        buf.sort(key=lambda x: x[0], reverse=True)
                        track_history[track_id]["crops_buffer"] = buf[:MAX_CROPS]
                
                # Fast OpenCV Color on sampled frame
                t0_color = time.time()
                x1, y1, x2, y2 = map(int, bbox)
                h, w, _ = frame.shape
                crop_full = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                
                if class_name.lower() in PERSON_CLASSES and crop_full.size > 0:
                    ch, cw, _ = crop_full.shape
                    shirt_crop = crop_full[int(0.1*ch):int(0.5*ch), int(0.2*cw):int(0.8*cw)]
                    color = detect_color_opencv(shirt_crop)
                    if color != "Unknown":
                        track_history[track_id]["opencv_colors"].append(color)
                elif class_name.lower() in VEHICLE_CLASSES and crop_full.size > 0:
                    color = detect_color_opencv(crop_full)
                    if color != "Unknown":
                        track_history[track_id]["opencv_colors"].append(color)
                        
                color_analysis_time += (time.time() - t0_color)
                
                
            if frame_idx % 30 == 0 or frame_idx == processor.total_frames - 1:
                print(f"Frame {frame_idx}/{processor.total_frames}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted.")
    finally:
        processor.release()

    # Lazy-load heavy models ONLY when needed in Pass 2
    extractor = None
    vehicle_classifier = None
    embedder = None

    final_detections = []
    sorted_tracks = sorted(track_history.values(), key=lambda x: x["first_seconds"])
    
    vehicle_class_count = 0
    person_attr_count = 0
    embedding_count = 0

    for track in sorted_tracks:
        avg_conf = sum(track["confidences"]) / len(track["confidences"]) if track["confidences"] else 0.0
        record = {
            "event_id": track["event_id"],
            "track_id": track["track_id"],
            "object": track["object"].capitalize(),
            "timestamp": track["first_seconds"],
            "frame": int(track["first_seconds"] * processor.fps),
            "confidence": round(avg_conf, 2),
            "bbox": track["smoothed_bbox"], # Using EMA smoothed box
            "clip_start": max(0, track["first_seconds"] - 3),
            "clip_end": track["last_seconds"] + 3,
            "speed": 0.0,
            "speed_unit": "Unknown",
            "action": "Unknown",
            "action_confidence": 0.0,
            "first_seen": track["first_seconds"],
            "last_seen": track["last_seconds"],
            "points": track.get("points", []),
            "average_speed": 0.0,
            "brand": "Unknown",
            "model": "Unknown",
            "brand_confidence": 0.0,
            "model_confidence": 0.0,
            "source_model": "yolo11",
            "track_uid": None,
            "query_concepts": [],
            "stationary": False,
            "attributes": {
                "shirt_color": "Unknown",
                "pants_color": "Unknown",
                "vehicle_color": "Unknown",
                "vehicle_type": "Unknown",
                "vehicle_make": "Unknown",
                "vehicle_model": "Unknown"
            }
        }
        
        # Speed & Action heuristics
        centers = track["centers"]
        if len(centers) > 5:
            dx = centers[-1][0] - centers[0][0]
            dy = centers[-1][1] - centers[0][1]
            dt = centers[-1][2] - centers[0][2]
            if dt > 0:
                px_per_sec = (dx**2 + dy**2)**0.5 / dt
                record["speed"] = round(px_per_sec, 2)
                record["average_speed"] = round(px_per_sec, 2)
                if px_per_sec < 50: record["speed_unit"] = "Slow"
                elif px_per_sec < 200: record["speed_unit"] = "Normal"
                else: record["speed_unit"] = "Fast"
                
                if record["object"].lower() in PERSON_CLASSES:
                    if px_per_sec < 20: record["action"], record["action_confidence"] = "Standing", 0.90
                    elif px_per_sec < 100: record["action"], record["action_confidence"] = "Walking", 0.85
                    else: record["action"], record["action_confidence"] = "Running", 0.88

        # Enrichment
        crops_buffer = track.get("crops_buffer", [])
        obj_lower = track["object"].lower()
        
        # We process the top 5 crops for multi-frame temporal voting
        if crops_buffer:
            intelligent_crops = [c[1] for c in crops_buffer]
            full_crops = [c[2] for c in crops_buffer]
            best_intelligent_crop = intelligent_crops[0]
            best_full_crop = full_crops[0]
            
            if obj_lower in PERSON_CLASSES:
                # Color Temporal Voting (OpenCV on intelligent shirt crops)
                colors = [detect_color_opencv(c) for c in intelligent_crops]
                fast_color = get_mode(colors)
                if fast_color != "Unknown":
                    record["attributes"]["shirt_color"] = fast_color
                    
                # Heavy feature extraction (Florence-2) on the absolute best full crop
                extractor = None
                if model_registry and "attribute_extractor" in model_registry:
                    extractor = model_registry["attribute_extractor"]
                else:
                    extractor = AttributeExtractor()
                shirt_desc = extractor.describe_crop(best_full_crop)
                florence_color = extractor.extract_color_from_text(shirt_desc, "Unknown")
                if florence_color != "Unknown":
                    record["attributes"]["shirt_color"] = florence_color
                person_attr_count += 1
                
            elif obj_lower in VEHICLE_CLASSES:
                # Color Temporal Voting
                colors = [detect_color_opencv(c) for c in intelligent_crops]
                fast_color = get_mode(colors)
                if fast_color != "Unknown":
                    record["attributes"]["vehicle_color"] = fast_color
                    
                # Vehicle Make/Model Multi-Frame Recognition
                if model_registry and "vehicle_classifier" in model_registry:
                    vehicle_classifier = model_registry["vehicle_classifier"]
                else:
                    if vehicle_classifier is None: vehicle_classifier = VehicleClassifier()
                brand, model, b_conf, m_conf = vehicle_classifier.classify_track(intelligent_crops)
                
                record["brand"] = brand
                record["model"] = model
                record["brand_confidence"] = round(b_conf, 3)
                record["model_confidence"] = round(m_conf, 3)
                record["attributes"]["vehicle_make"] = brand
                record["attributes"]["vehicle_model"] = model
                vehicle_class_count += 1

        # Description
        desc_parts = []
        if record["attributes"].get("vehicle_color") != "Unknown":
            desc_parts.append(record["attributes"]["vehicle_color"])
        if record["attributes"].get("shirt_color") != "Unknown":
            desc_parts.append(record["attributes"]["shirt_color"] + " shirt")
            
        if record.get("brand") and record["brand"] != "Unknown":
            desc_parts.append(record["brand"])
            if record.get("model") and record["model"] != "Unknown":
                desc_parts.append(record["model"])
        else:
            desc_parts.append(record["object"])
            
        record["description"] = " ".join(desc_parts).capitalize()
        
        # Embeddings
        if model_registry and "embedder" in model_registry:
            embedder = model_registry["embedder"]
        else:
            if embedder is None: embedder = Embedder()
        
        text_emb = embedder.embed_text(record["description"])
        
        best_crop_for_emb = crops_buffer[0][2] if crops_buffer else None
        img_emb = embedder.embed_crop(best_crop_for_emb) if best_crop_for_emb is not None else None
        
        if img_emb is not None and text_emb is not None:
            combined = (img_emb + text_emb) / 2.0
            record["embedding"] = (combined / np.linalg.norm(combined)).tolist()
            embedding_count += 1
        elif img_emb is not None:
            record["embedding"] = img_emb.tolist()
            embedding_count += 1
        elif text_emb is not None:
            record["embedding"] = text_emb.tolist()
            embedding_count += 1
        else:
            record["embedding"] = None
            
        final_detections.append(record)

    # ---------------------------------------------------------
    # CANDIDATE INCIDENT ANALYSIS & EVENT FUSION
    # ---------------------------------------------------------
    print("==================================================")
    print("Running Candidate Incident Analysis...")
    
    if model_registry and "temporal_analyzer" in model_registry:
        temporal_analyzer = model_registry["temporal_analyzer"]
    else:
        try:
            from mmaction_analyzer import TemporalActionAnalyzer
            temporal_analyzer = TemporalActionAnalyzer()
        except Exception as e:
            print(f"Warning: TemporalActionAnalyzer unavailable: {e}")
            temporal_analyzer = None

    incident_events = []
    
    # Pre-filter candidate pairs to avoid O(N^2) explosion
    candidate_pairs = []
    
    # Metrics logging
    perf = {
        "total_tracks": len(final_detections),
        "candidates_considered": 0,
        "rejected_class": 0,
        "rejected_time": 0,
        "rejected_distance": 0,
        "rejected_trajectory": 0,
        "rejected_motion": 0,
        "temporal_evals": 0,
        "incidents_created": 0,
        "merged": 0
    }
    
    # 1. Gather all potential interacting pairs
    for i in range(len(final_detections)):
        for j in range(i + 1, len(final_detections)):
            perf["candidates_considered"] += 1
            t1, t2 = final_detections[i], final_detections[j]

            # Incident generation is deliberately YOLO11-only. Arbitrary
            # YOLOE evidence is persisted and searchable but cannot create or
            # strengthen accident/fight candidates.
            if (t1.get("source_model") or "yolo11") != "yolo11" or (t2.get("source_model") or "yolo11") != "yolo11":
                perf["rejected_class"] += 1
                continue
            
            # A. Time Overlap Gate
            overlap_start = max(t1.get("first_seen", 0), t2.get("first_seen", 0))
            overlap_end = min(t1.get("last_seen", 0), t2.get("last_seen", 0))
            
            if overlap_end <= overlap_start:
                perf["rejected_time"] += 1
                continue
                
            # B. Class Gate
            obj1, obj2 = t1["object"].lower(), t2["object"].lower()
            is_person_vehicle = (obj1 in PERSON_CLASSES and obj2 in VEHICLE_CLASSES) or (obj2 in PERSON_CLASSES and obj1 in VEHICLE_CLASSES)
            is_multi_vehicle = (obj1 in VEHICLE_CLASSES and obj2 in VEHICLE_CLASSES)
            is_multi_person = (obj1 in PERSON_CLASSES and obj2 in PERSON_CLASSES)
            
            if not (is_person_vehicle or is_multi_vehicle or is_multi_person):
                perf["rejected_class"] += 1
                continue
                
            # Define event type base on interaction
            if is_multi_vehicle or is_person_vehicle:
                event_type = "possible_accident"
            elif is_multi_person:
                event_type = "possible_fight"
            else:
                perf["rejected_class"] += 1
                continue
                
            # C. Frame-by-frame Distance & IoU analysis
            p1_map = {p["frame"]: p for p in t1.get("points", [])}
            p2_map = {p["frame"]: p for p in t2.get("points", [])}
            common_frames = sorted(list(set(p1_map.keys()).intersection(set(p2_map.keys()))))
            
            if not common_frames:
                perf["rejected_time"] += 1
                continue
                
            min_dist = float('inf')
            max_iou = 0.0
            closest_frame = common_frames[0]
            
            for f in common_frames:
                pt1 = p1_map[f]
                pt2 = p2_map[f]
                d = calculate_distance(pt1["bbox"], pt2["bbox"])
                io = calculate_iou(pt1["bbox"], pt2["bbox"])
                
                if d < min_dist:
                    min_dist = d
                    closest_frame = f
                if io > max_iou:
                    max_iou = io

            # D. Motion/Trajectory & Sudden Deceleration Analysis
            decel_1, decel_2 = False, False
            
            def get_speed_near(pt_map, target_frame, offset):
                f = target_frame + offset
                if f in pt_map:
                    return pt_map[f].get("speed", 0.0)
                # Fallback
                for o in range(offset, offset + (5 if offset > 0 else -5), 1 if offset > 0 else -1):
                    if target_frame + o in pt_map:
                        return pt_map[target_frame + o].get("speed", 0.0)
                return None

            s1_before = get_speed_near(p1_map, closest_frame, -5)
            s1_after = get_speed_near(p1_map, closest_frame, 5)
            s2_before = get_speed_near(p2_map, closest_frame, -5)
            s2_after = get_speed_near(p2_map, closest_frame, 5)
            
            if s1_before is not None and s1_after is not None:
                if s1_before > 40 and s1_after < s1_before * 0.5:
                    decel_1 = True
            if s2_before is not None and s2_after is not None:
                if s2_before > 40 and s2_after < s2_before * 0.5:
                    decel_2 = True
                    
            has_deceleration = decel_1 or decel_2
            has_fast_obj = (t1.get("speed_unit") == "Fast") or (t2.get("speed_unit") == "Fast") or (any(p.get("speed", 0) > 100 for p in t1.get("points", []))) or (any(p.get("speed", 0) > 100 for p in t2.get("points", [])))
            
            interaction_score = min(1.0, overlap_end - overlap_start)
            proximity_score = min(1.0, 100 / max(min_dist, 1)) if max_iou == 0 else min(1.0, max_iou * 2 + 0.5)
            motion_score = 1.0 if (has_fast_obj or has_deceleration) else 0.5
            trajectory_score = 1.0 if max_iou > 0.05 else (0.8 if min_dist < 80 else 0.4)
            
            # Gating based on frame-by-frame proximity
            if event_type == "possible_accident":
                if min_dist > 150 and max_iou < 0.02:
                    perf["rejected_distance"] += 1
                    continue
                if not (has_fast_obj or has_deceleration) and max_iou < 0.05:
                    perf["rejected_motion"] += 1
                    continue
            elif event_type == "possible_fight":
                if min_dist > 80 or max_iou == 0:
                    perf["rejected_distance"] += 1
                    continue
                if interaction_score < 0.5:
                    perf["rejected_time"] += 1
                    continue

            closest_time = p1_map[closest_frame]["timestamp"]
            
            geom_score = (interaction_score + proximity_score) / 2.0
            motion_traj = (motion_score + trajectory_score) / 2.0
            cand_score = (0.50 * geom_score) + (0.50 * motion_traj)
            
            if cand_score < 0.55:
                perf["rejected_motion"] += 1
                continue
                
            candidate_pairs.append({
                "t1": t1,
                "t2": t2,
                "mid_time": closest_time,
                "event_type": event_type,
                "interaction_score": interaction_score,
                "proximity_score": proximity_score,
                "motion_score": motion_score,
                "trajectory_score": trajectory_score,
                "has_deceleration": has_deceleration,
                "min_dist": min_dist,
                "max_iou": max_iou
            })
            
    # 2. Run Temporal Model ONLY on survivors
    for cand in candidate_pairs:
        t1, t2 = cand["t1"], cand["t2"]
        mid_time = cand["mid_time"]
        event_type = cand["event_type"]
        
        print(f"Valid Candidate found: {event_type} at {mid_time:.1f}s (min_dist={cand['min_dist']:.1f}px, max_iou={cand['max_iou']:.3f})")
        action_score = 0.0
        action_label = "none"
        reason = [
            f"proximity score {cand['proximity_score']:.2f}",
            f"trajectory score {cand['trajectory_score']:.2f}"
        ]
        if cand["has_deceleration"]:
            reason.append("sudden deceleration detected")
        
        if temporal_analyzer:
            perf["temporal_evals"] += 1
            clip_start = max(0, mid_time - 3)
            clip_end = min(processor.duration, mid_time + 3)
            analysis = temporal_analyzer.analyze_clip(source, clip_start, clip_end)
            
            if analysis.get("actions"):
                top_action = analysis["actions"][0]
                action_label = top_action["label"].lower()
                raw_temporal = top_action["score"]
                
                if event_type == "possible_accident":
                    generic_actions = ["driving car", "motorcycling", "walking", "running", "riding scooter", "parking"]
                    if any(a in action_label for a in generic_actions):
                        action_score = min(0.15, raw_temporal)
                    else:
                        action_score = raw_temporal
                elif event_type == "possible_fight":
                    if "fight" in action_label or "wrestl" in action_label or "punch" in action_label:
                        action_score = raw_temporal
                    else:
                        action_score = min(0.15, raw_temporal)
                else:
                    action_score = raw_temporal
                
                reason.append(f"temporal model support ({action_label})")
                
        # 3. Final Fusion (Weighted Model)
        geom_score = (cand["interaction_score"] + cand["proximity_score"]) / 2.0
        motion_traj = (cand["motion_score"] + cand["trajectory_score"]) / 2.0
        
        final_confidence = (0.45 * geom_score) + (0.45 * motion_traj) + (0.10 * action_score)
        
        # Calculate physical evidence strength
        physical_strength = (geom_score + motion_traj) / 2.0
        
        is_strong_physical = (cand["proximity_score"] > 0.80 and cand["trajectory_score"] > 0.80)
        is_decent_physical = physical_strength > 0.65
        
        status = None
        
        # Two-level incident system
        if final_confidence >= 0.70 and action_score > 0.3:
            status = "CONFIRMED"
        elif is_strong_physical or (is_decent_physical and final_confidence >= 0.50):
            status = "POSSIBLE"
            
        if status:
            incident_id = f"evt_{uuid.uuid4().hex[:8]}"
            video_name_val = processor.video_path.split("/")[-1]
            
            # Setup evidence breakdown payload exactly as requested
            evidence_data = {
                "iou": round(cand["max_iou"], 3),
                "minimum_distance": round(cand["min_dist"], 1),
                "relative_velocity": 0.0, # Placeholder unless calculated
                "deceleration": cand["has_deceleration"],
                "trajectory_intersection": cand["trajectory_score"] > 0.5,
                "interaction_duration": round(cand["interaction_score"] * 5.0, 2), # proxy for duration
                "videomae_action": action_label,
                "videomae_confidence": round(action_score, 3)
            }
            
            evidence_breakdown = {
                "status": status,
                "geometry_score": round(geom_score, 3),
                "motion_score": round(motion_traj, 3),
                "temporal_score": round(action_score, 3),
                "final_confidence": round(final_confidence, 3),
                "involved_tracks": [t1["track_id"], t2["track_id"]],
                "evidence": evidence_data
            }
            
            inc_event = {
                "event_id": incident_id,
                "video_id": video_id if 'video_id' in locals() else video_name_val,
                "video_name": video_name_val,
                "camera_id": "CAM_01",
                "location": "Unknown",
                "frame": int(mid_time * processor.fps),
                "timestamp": round(mid_time, 2),
                "track_id": -1,
                "track_uid": None,
                "object": f"Incident: {event_type.replace('_', ' ').capitalize()}",
                "action": action_label,
                "action_confidence": round(action_score, 3),
                "confidence": round(final_confidence, 3),
                "bbox": [],
                "attributes": {},
                "speed": None,
                "speed_unit": None,
                "clip_start": max(0.0, round(mid_time - 3.0, 2)),
                "clip_end": round(mid_time + 3.0, 2),
                "description": f"Detected {status.lower()} {event_type.replace('_', ' ')} involving {t1['object']} and {t2['object']}",
                "event_type": event_type,
                "sub_event_type": action_label,
                "incident_confidence": round(final_confidence, 3),
                "reason": evidence_breakdown, # Store the full object in reason
                "involved_track_ids": [t1["track_id"], t2["track_id"]]
                ,"source_model": "yolo11"
            }
            incident_events.append(inc_event)
            perf["incidents_created"] += 1
            
    # 4. Deduplication
    filtered_incidents = []
    for inc in incident_events:
        is_duplicate = False
        for f_inc in filtered_incidents:
            # Check 5 second window and same event type
            if inc["event_type"] == f_inc["event_type"] and abs(inc["timestamp"] - f_inc["timestamp"]) <= 5.0:
                # Merge track IDs
                merged_tracks = list(set(f_inc.get("involved_track_ids", []) + inc.get("involved_track_ids", [])))
                f_inc["involved_track_ids"] = merged_tracks
                f_inc["description"] = f"Detected {f_inc['event_type'].replace('_', ' ')} involving multiple objects"
                
                if inc["confidence"] > f_inc["confidence"]:
                    f_inc["confidence"] = inc["confidence"]
                    f_inc["incident_confidence"] = inc["incident_confidence"]
                    
                is_duplicate = True
                perf["merged"] += 1
                break
                
        if not is_duplicate:
            # Embed text for the final unique event
            if model_registry and "embedder" in model_registry:
                embedder = model_registry["embedder"]
            else:
                if 'embedder' not in locals() or embedder is None: 
                    embedder = Embedder()
                    
            text_emb = embedder.embed_text(inc["description"])
            inc["embedding"] = text_emb.tolist() if text_emb is not None else None
            
            filtered_incidents.append(inc)

    final_detections.extend(filtered_incidents)
    
    # 5. Performance Summary Output
    print("\n==================================================")
    print("INCIDENT ANALYSIS SUMMARY")
    print("==================================================")
    print(f"Total tracks: {perf['total_tracks']}")
    print(f"Candidate pairs: {perf['candidates_considered']}")
    print(f"Rejected by class: {perf['rejected_class']}")
    print(f"Rejected by time: {perf['rejected_time']}")
    print(f"Rejected by distance: {perf['rejected_distance']}")
    print(f"Rejected by trajectory/motion: {perf['rejected_motion']}")
    print(f"Temporal clips evaluated: {perf['temporal_evals']}")
    print(f"Possible incidents: {perf['incidents_created']}")
    print(f"Merged duplicates: {perf['merged']}")
    print(f"Final incidents: {len(filtered_incidents)}")
    print("==================================================\n")

    # Summarize colors and objects for the terminal output
    print("Color analysis:")
    color_counts = {}
    person_count = 0
    for t in final_detections:
        obj = t["object"].lower()
        if obj in PERSON_CLASSES:
            person_count += 1
        elif obj in VEHICLE_CLASSES:
            c = t["attributes"].get("vehicle_color", "Unknown")
            if c != "Unknown":
                key = f"{c} cars"
                color_counts[key] = color_counts.get(key, 0) + 1
    
    for k, v in color_counts.items():
        print(f"{k}: {v}")
    print(f"persons: {person_count}")
    
    print(f"Events created: {len(final_detections)} (including {len(filtered_incidents)} incidents)")
    print("Generating embeddings...")
    print(f"Embeddings created: {embedding_count}")
    print(f"FAISS vectors: {embedding_count}")
    print(f"DATABASE EVENTS: {len(final_detections)}")
    print("ANALYSIS COMPLETE")

    return {
        "video_duration": processor.duration,
        "detections": final_detections
    }
