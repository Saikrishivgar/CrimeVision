"""Central configuration for the CrimeVision detector branches.

YOLO11 settings intentionally remain compatible with the existing pipeline.
YOLOE settings are conservative defaults and can be overridden with
environment variables while benchmark data is collected.
"""

import os


YOLO_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO11_MODEL_PATH = os.path.join(YOLO_DIR, "yolo11n.pt")
YOLOE_MODEL_PATH = os.environ.get(
    "CRIMEVISION_YOLOE_MODEL_PATH",
    os.path.join(YOLO_DIR, "yoloe-11m-seg.pt"),
)

# Existing YOLO11 sampling remains unchanged.
PRIMARY_FRAME_STRIDE = int(os.environ.get("CRIMEVISION_PRIMARY_FRAME_STRIDE", "10"))

# Open-vocabulary inference is opt-in and intentionally configurable.
OPEN_VOCAB_FRAME_STRIDE = int(os.environ.get("CRIMEVISION_OPEN_VOCAB_FRAME_STRIDE", "10"))
OPEN_VOCAB_CONFIDENCE_THRESHOLD = float(
    os.environ.get("CRIMEVISION_OPEN_VOCAB_CONFIDENCE_THRESHOLD", "0.15")
)
OPEN_VOCAB_MIN_HITS = int(os.environ.get("CRIMEVISION_OPEN_VOCAB_MIN_HITS", "3"))
OPEN_VOCAB_MIN_TRACK_DURATION = float(
    os.environ.get("CRIMEVISION_OPEN_VOCAB_MIN_TRACK_DURATION", "0.4")
)
OPEN_VOCAB_MAX_GAP_SECONDS = float(
    os.environ.get("CRIMEVISION_OPEN_VOCAB_MAX_GAP_SECONDS", "2.0")
)
OPEN_VOCAB_MIN_BBOX_IOU = float(
    os.environ.get("CRIMEVISION_OPEN_VOCAB_MIN_BBOX_IOU", "0.05")
)
OPEN_VOCAB_STATIONARY_SPEED = float(
    os.environ.get("CRIMEVISION_OPEN_VOCAB_STATIONARY_SPEED", "8.0")
)

YOLOE_TRACKER = os.environ.get("CRIMEVISION_YOLOE_TRACKER", "botsort.yaml")
YOLOE_SOURCE_MODEL = "yoloe"
YOLO11_SOURCE_MODEL = "yolo11"

# Grounding DINO is an optional *verification* step for stable YOLOE tracks.
# It is never initialized during normal YOLO11 analysis.  The default model is
# configurable so CPU-only machines can use ``grounding-dino-tiny`` after
# benchmarking if the base checkpoint is not practical for their footage.
GROUNDING_DINO_ENABLED = os.environ.get("CRIMEVISION_GROUNDING_DINO_ENABLED", "true").lower() == "true"
GROUNDING_DINO_MODEL_ID = os.environ.get(
    "CRIMEVISION_GROUNDING_DINO_MODEL_ID",
    "IDEA-Research/grounding-dino-base",
)
GROUNDING_DINO_DEVICE = os.environ.get("CRIMEVISION_GROUNDING_DINO_DEVICE", "auto")

# The base box/text defaults follow the published Transformers Grounding DINO
# inference example.  Identity thresholds below are deliberately centralized
# so they can be calibrated against the local CCTV benchmark rather than baked
# into the detector code.
GROUNDING_DINO_BOX_THRESHOLD = float(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_BOX_THRESHOLD", "0.40")
)
GROUNDING_DINO_TEXT_THRESHOLD = float(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_TEXT_THRESHOLD", "0.30")
)
GROUNDING_DINO_MAX_SAMPLES_PER_TRACK = int(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_MAX_SAMPLES_PER_TRACK", "5")
)
GROUNDING_DINO_MIN_VERIFIED_OBSERVATIONS = int(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_MIN_VERIFIED_OBSERVATIONS", "2")
)
GROUNDING_DINO_MIN_VERIFIED_SCORE = float(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_MIN_VERIFIED_SCORE", "0.55")
)
GROUNDING_DINO_MIN_VOTE_RATIO = float(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_MIN_VOTE_RATIO", "0.70")
)
GROUNDING_DINO_MIN_VOTE_MARGIN = float(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_MIN_VOTE_MARGIN", "0.15")
)
GROUNDING_DINO_YOLOE_WEIGHT = float(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_YOLOE_WEIGHT", "0.30")
)
GROUNDING_DINO_WEIGHT = float(
    os.environ.get("CRIMEVISION_GROUNDING_DINO_WEIGHT", "0.70")
)
