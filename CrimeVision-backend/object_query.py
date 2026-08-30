"""Natural-language object normalization for YOLO11/YOLOE routing."""

import re


OBJECT_CONCEPTS = {
    "speaker": ["speaker", "loudspeaker", "audio speaker", "specker", "speeker", "spk", "spkr", "woofer", "portable speaker", "bluetooth speaker", "small speaker", "red speaker"],
    "computer": [
        "desktop computer",
        "PC tower",
        "computer case",
        "computer chassis",
        "computer cabinet",
        "PC cabinet",
        "CPU",
        "computer",
        "pc",
        "cabinet",
    ],
    "monitor": ["computer monitor", "display screen", "monitor", "screen", "display"],
    "laptop": ["laptop", "notebook computer", "notebook"],
    "phone": ["smartphone", "mobile phone", "cell phone", "mobile", "telephone"],
    "keyboard": ["keyboard", "computer keyboard", "key board"],
    "mouse": ["computer mouse", "mouse"],
    "printer": ["printer", "office printer"],
    "router": ["router", "wifi router", "network router"],
    "cctv camera": ["CCTV camera", "security camera", "surveillance camera", "camera", "cctv"],
    "television": ["television", "TV", "television screen", "tv screen"],
    "tablet": ["tablet", "tablet computer", "ipad"],
    "projector": ["projector", "video projector"],
    "headphones": ["headphones", "headset", "earphones", "earphone"],
}

ELECTRONIC_OBJECTS = [
    "speaker",
    "computer",
    "monitor",
    "laptop",
    "keyboard",
    "mouse",
    "printer",
    "router",
    "cctv camera",
    "television",
    "phone",
    "tablet",
    "projector",
    "headphones",
]

# Common natural-language forms for classes already handled by YOLO11. These
# must be recognized before the generic-object fallback; otherwise phrases
# such as "any person detected" can be mistaken for arbitrary YOLOE objects.
YOLO11_OBJECT_ALIASES = {
    "person": ["person", "people", "man", "woman", "guy", "individual", "pedestrian"],
    "car": ["car", "cars", "vehicle", "vehicles", "sedan", "suv"],
    "motorcycle": ["motorcycle", "motorcycles", "bike", "bikes", "scooter", "scooters"],
    "bus": ["bus", "buses"],
    "truck": ["truck", "trucks", "lorry", "lorries"],
    "bicycle": ["bicycle", "bicycles", "cycle", "cycles"],
    "backpack": ["backpack", "backpacks"],
    "suitcase": ["suitcase", "suitcases", "luggage"],
    "chair": ["chair", "chairs"],
    "television": ["television", "televisions", "tv", "tvs"],
}

# These are served accurately by the existing YOLO11/COCO route for ordinary
# searches. Electronics listed in OBJECT_CONCEPTS are deliberately preferred
# for YOLOE when explicitly requested, even where COCO has a related class.
YOLO11_NATIVE = {
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "backpack",
    "suitcase",
    "chair",
    "laptop",
    "keyboard",
    "phone",
    "cell phone",
    "mouse",
    "television",
}

YOLOE_PREFERRED = set(OBJECT_CONCEPTS) | {"laptop", "keyboard", "mouse", "phone"}

# These describe events or situations, not physical objects. Keep them out of
# the generic-object fallback so a query such as "any accident detected" does
# not become an open-vocabulary YOLOE request for the phrase itself.
INCIDENT_QUERY_TERMS = (
    "accident",
    "collision",
    "crash",
    "incident",
    "fight",
    "violence",
    "fall",
    "fell down",
    "robbery",
    "stealing",
    "suspicious",
)

COLORS = [
    "red",
    "blue",
    "black",
    "white",
    "green",
    "yellow",
    "orange",
    "grey",
    "gray",
    "silver",
    "brown",
    "purple",
]


def _canonicalize(value):
    value = re.sub(r"\s+", " ", value.lower().strip())
    if value.endswith("ies"):
        value = value[:-3] + "y"
    elif value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def _concept_key_from_text(text):
    lowered = text.lower()
    # Longest phrases first so "pc tower" wins over "computer".
    aliases = []
    for key, concepts in OBJECT_CONCEPTS.items():
        aliases.append((key, key))
        aliases.extend((concept, key) for concept in concepts)
    for phrase, key in sorted(aliases, key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase.lower())}\b", lowered):
            return key
    return None


def _extract_generic_object(text):
    cleaned = text.lower()
    cleaned = re.sub(
        r"\b(find|show|track|locate|detect|where is|where did|how many|search for|look for)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\b(all|the|a|an|this|that|in|on|from|video|videos)\b", " ", cleaned)
    for color in COLORS:
        cleaned = re.sub(rf"\b{color}\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 -]", " ", cleaned)
    candidate = re.sub(r"\s+", " ", cleaned).strip()
    if any(re.search(rf"\b{re.escape(term)}\b", candidate) for term in INCIDENT_QUERY_TERMS):
        return None
    if candidate in {
        "",
        "object",
        "objects",
        "electronic devices",
        "everything",
        "fully map",
        "fully mapped",
        "map this",
        "map this video",
        "all detections",
        "all vehicles and people",
    }:
        return None
    return _canonicalize(candidate)


def extract_object_requests(text):
    """Return all requested canonical objects, including multi-object map asks."""
    lowered = text.lower()
    requests = []
    if "electronic device" in lowered or "electronic equipment" in lowered:
        requests.extend(ELECTRONIC_OBJECTS)

    for key, concepts in OBJECT_CONCEPTS.items():
        phrases = [key] + concepts
        if any(re.search(rf"\b{re.escape(p.lower())}s?\b", lowered) for p in phrases):
            if key not in requests:
                requests.append(key)

    # Resolve native YOLO11 classes before falling back to arbitrary text.
    # This keeps ordinary surveillance questions on YOLO11 even when the
    # Extended Object Detection control is enabled.
    for key, phrases in YOLO11_OBJECT_ALIASES.items():
        if any(re.search(rf"\b{re.escape(p.lower())}\b", lowered) for p in phrases):
            if key not in requests:
                requests.append(key)

    if not requests:
        candidate = _extract_generic_object(text)
        if candidate:
            requests.append(candidate)
    return requests


def normalize_object_request(text):
    text = text or ""
    lowered = text.lower()
    object_requests = extract_object_requests(text)
    object_name = object_requests[0] if object_requests else None

    if object_name in OBJECT_CONCEPTS:
        concepts = OBJECT_CONCEPTS[object_name]
    elif object_name:
        concepts = [object_name]
    else:
        concepts = []

    color = next((c for c in COLORS if re.search(rf"\b{c}\b", lowered)), None)
    normalized_color = "gray" if color == "grey" else color
    is_open_vocab = bool(
        object_name
        and (object_name not in YOLO11_NATIVE or object_name in YOLOE_PREFERRED)
    )

    return {
        "object": object_name,
        "objects": object_requests,
        "concepts": concepts,
        "color": normalized_color,
        "requires_open_vocab": is_open_vocab,
        "intent": "object_search" if object_name else None,
    }
