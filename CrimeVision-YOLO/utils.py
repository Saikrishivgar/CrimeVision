import cv2

def draw_annotations(frame, detections):
    """
    Draws bounding boxes, track IDs, class names, and attributes on the frame.
    
    Args:
        frame: OpenCV image frame.
        detections: list of dicts from the inference pipeline.
        
    Returns:
        frame: OpenCV image frame with drawings.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']}"
        if det["track_id"] is not None:
            label += f" #{det['track_id']}"
        label += f" {det['confidence']:.2f}"
        
        # Color coding: Red for weapons/hazards, Green for others, Orange for fire
        color = (0, 255, 0)  # Green
        cls_lower = det["class"].lower()
        if cls_lower in ["knife", "gun"]:
            color = (0, 0, 255)  # Red
        elif cls_lower in ["fire", "smoke"]:
            color = (0, 165, 255)  # Orange
            
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Extra details (attributes)
        attrs = det.get("attributes", {})
        attr_label = ""
        if "shirt_color" in attrs and attrs["shirt_color"] != "Unknown":
            attr_label += f"Shirt: {attrs['shirt_color']} "
        if "pant_color" in attrs and attrs["pant_color"] != "Unknown":
            attr_label += f"Pant: {attrs['pant_color']}"
        if "vehicle_color" in attrs and attrs["vehicle_color"] != "Unknown":
            attr_label += f"Color: {attrs['vehicle_color']}"
            
        # Draw label background
        lbl_y = max(15, y1 - 25)
        
        # Main label (class + id + confidence)
        cv2.putText(frame, label, (x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Attribute sub-label (if attributes exist)
        if attr_label.strip():
            cv2.putText(
                frame, 
                attr_label.strip(), 
                (x1, max(15, y1 - 7)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )

    return frame

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) for two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def calculate_distance(boxA, boxB):
    """Calculates Euclidean distance between centers of two boxes."""
    cxA, cyA = calculate_center(boxA)
    cxB, cyB = calculate_center(boxB)
    import math
    return math.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)
