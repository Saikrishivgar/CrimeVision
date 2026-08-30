import json
import re

from object_query import normalize_object_request

def optimize_prompt(user_text, context=None):
    """
    Simulates an LLM converting a natural language query into a structured filter,
    taking into account previous conversation context.
    """

    user_text = user_text.lower()
    filters = {}
    
    # 1. Temporal parsing
    # "What happened at 20 seconds?" -> time_start: 19, time_end: 21
    time_match = re.search(r'at (\d+) seconds?', user_text)
    if time_match:
        t = int(time_match.group(1))
        filters["time_start"] = max(0, t - 2)
        filters["time_end"] = t + 2
        
    time_around = re.search(r'around (\d+) seconds?', user_text)
    if time_around:
        t = int(time_around.group(1))
        filters["time_start"] = max(0, t - 3)
        filters["time_end"] = t + 3
        
    time_between = re.search(r'between (\d+) and (\d+) seconds?', user_text)
    if time_between:
        t1 = int(time_between.group(1))
        t2 = int(time_between.group(2))
        filters["time_start"] = t1
        filters["time_end"] = t2
    
    # 2. Context handling (e.g., "when did it first appear?")
    if ("it" in user_text or "that" in user_text or "this" in user_text) and context:
        # inherit filters from context
        filters.update(context)
        
    if "first" in user_text:
        filters["sort"] = "first"
        
    # Intents
    if "why" in user_text:
        filters["intent"] = "explain"
    elif "show me that" in user_text or "play that" in user_text or "show that" in user_text:
        filters["intent"] = "show_evidence"
    elif "how long" in user_text:
        filters["intent"] = "duration"
        
    # Track ID detection (e.g. "motorcycle #28" or "track 28")
    track_match = re.search(r'(?:#|track\s*)(\d+)', user_text)
    if track_match:
        filters["track_id"] = int(track_match.group(1))
        filters["intent"] = "seek_map"
        
    # Check if seeking an incident
    if any(k in user_text for k in ["show me the accident", "show accident", "show the collision", "show the fight", "show the incident"]):
        filters["intent"] = "seek_map"
        
    # Check if seeking a time (if time was found and 'what happened' or 'show' is used)
    if ("what happened" in user_text or "show" in user_text) and "time_start" in filters and "track_id" not in filters:
        filters["intent"] = "seek_map"
    
    # 3. Detect objects
    vehicles = ["car", "suv", "truck", "bike", "motorcycle", "vehicle", "sedan", "bus"]
    persons = ["person", "man", "woman", "guy", "suspect", "individual", "people"]
    
    # Check for exact matches first to preserve "truck" or "motorcycle"
    for v in vehicles:
        if v in user_text:
            filters["object"] = v if v != "vehicle" else "vehicle"
            break
            
    if "object" not in filters:
        if any(p in user_text for p in persons):
            filters["object"] = "person"

    # Open-vocabulary object normalization. This is additive to the existing
    # vehicle/person parser and does not change the normal YOLO11 route.
    object_request = normalize_object_request(user_text)
    if object_request.get("object"):
        if object_request["requires_open_vocab"] or "object" not in filters:
            filters["object"] = object_request["object"]
        filters["objects"] = object_request.get("objects", [])
        filters["object_concepts"] = object_request.get("concepts", [])
        filters["requires_open_vocab"] = object_request.get("requires_open_vocab", False)
        filters["query_attribute"] = object_request.get("color")
            
    # 4. Detect colors
    colors = ["red", "blue", "black", "white", "green", "yellow", "orange", "grey", "gray", "silver"]
    for c in colors:
        if re.search(rf"\b{c}\b", user_text):
            filters["color"] = c
            break

    # 4.5 Detect Brands
    brands = [
        "toyota", "hyundai", "honda", "maruti suzuki", "tata", "mahindra", 
        "kia", "bmw", "mercedes-benz", "audi", "ford", "volkswagen", "chevrolet",
        "nissan", "renault", "skoda", "jeep", "mg", "volvo", "lexus", "porsche", "land rover"
    ]
    
    for b in brands:
        if b in user_text:
            filters["brand"] = b
            filters["object"] = "vehicle" # Implicitly it's a vehicle
            break

    # 5. Detect actions
    actions = ["passing", "walking", "running", "parked"]
    for a in actions:
        if a in user_text:
            filters["action"] = a
            break
            
    # 6. Detect Incident Events
    if any(word in user_text for word in ["accident", "collision", "crash"]):
        filters["event_type"] = "possible_accident"
    elif any(word in user_text for word in ["fall", "fell down"]):
        filters["event_type"] = "person_fall"
    elif any(word in user_text for word in ["fight", "violence"]):
        filters["event_type"] = "possible_fight"
    elif any(word in user_text for word in ["suspicious", "robbery", "stealing"]):
        filters["event_type"] = "possible_robbery"
    elif any(word in user_text for word in ["weapon", "gun", "knife", "armed"]):
        filters["event_type"] = "possible_weapon_incident"
            
    # Normalize description for embedding (SigLIP/CLIP)
    description = ""
    if filters.get("color"):
        description += filters["color"] + " "
    if filters.get("object") == "person":
        description += "person "
    elif filters.get("object") == "vehicle":
        description += "car "
    elif filters.get("object"):
        description += filters["object"] + " "
    if filters.get("action"):
        description += filters["action"]
        
    filters["normalized_query"] = description.strip() or user_text
    
    return filters

def generate_fir_text(detections, video_id):
    """
    Simulates an LLM generating an incident report from the detection timeline.
    """
    if not detections:
        return f"### INITIAL INCIDENT REPORT (FIR)\n\n**Video ID**: {video_id}\n\n**Summary**: No significant entities or threats were detected in this footage."
        
    report = f"### INITIAL INCIDENT REPORT (FIR)\n\n"
    report += f"**Video Source**: {video_id}\n"
    report += f"**Date of Analysis**: Auto-Generated\n\n"
    
    report += "**Summary of Events**:\n"
    report += f"The AI analysis pipeline processed the footage and detected {len(detections)} notable tracking events.\n\n"
    
    report += "**Evidence Timeline**:\n"
    
    # Sort by time
    sorted_det = sorted(detections, key=lambda x: x.get('timestamp', 0))
    
    for idx, d in enumerate(sorted_det[:10]):
        time = f"{d.get('timestamp', 0):.1f}s"
        obj = d.get('object', 'Unknown')
        
        attrs = d.get('attributes', {})
        details = []
        if attrs.get('shirt_color') and attrs.get('shirt_color') != 'Unknown':
            details.append(f"{attrs['shirt_color']} shirt")
        if attrs.get('vehicle_color') and attrs.get('vehicle_color') != 'Unknown':
            details.append(f"{attrs['vehicle_color']} vehicle")
            
        detail_str = f" ({', '.join(details)})" if details else ""
            
        report += f"At **{time}**, a {obj.upper()}{detail_str} was identified.\n"
        
    if len(detections) > 10:
        report += f"\n*...and {len(detections) - 10} more events not listed in the summary.*\n"
        
    report += "\n**Recommended Action**: Archive footage and attach to case file."
    
    return report
