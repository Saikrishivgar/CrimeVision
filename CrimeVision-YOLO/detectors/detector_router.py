"""Routing helpers for primary versus optional open-vocabulary detection."""


def should_use_yoloe(filters=None, extended_enabled=False):
    """Return whether the optional branch is needed for this request.

    Native YOLO11 object searches must stay on the existing detector even if
    the UI toggle is enabled. The toggle enables YOLOE for extended mapping,
    while an explicit open-vocabulary filter always enables it.
    """
    filters = filters or {}
    if filters.get("requires_open_vocab"):
        return True
    # Incident/event queries are answered from persisted event data and the
    # YOLO11 analysis pipeline. An enabled YOLOE toggle must not reinterpret
    # a missing object field as an open-vocabulary object request.
    if filters.get("event_type"):
        return False
    if not extended_enabled:
        return False
    return filters.get("intent") in {"full_mapping", "extended_mapping"} or not filters.get("object")
