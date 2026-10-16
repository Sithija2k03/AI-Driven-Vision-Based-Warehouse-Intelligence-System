# confidence_filter.py
# Novelty 1 — Keypoint Confidence Filter with Body-Region Freeze State
# F3 Pipeline | Warehouse Intelligence System | J26-DS-352

# YOLOv8m-pose keypoint indices (17 keypoints)
# 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip
# 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

CONFIDENCE_THRESHOLD = 0.5
FREEZE_RATIO = 0.5   # if more than 50% of a zone's keypoints are low-conf → freeze

BODY_ZONES = {
    "upper": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # head, shoulders, arms, wrists
    "core":  [5, 6, 11, 12],                          # shoulders + hips (torso)
    "lower": [11, 12, 13, 14, 15, 16]                 # hips, knees, ankles
}

# Holds the last known good keypoints per worker
_freeze_store = {}


def check_confidence(keypoints, worker_id):
    """
    keypoints : list of 17 items, each is [x, y, confidence]
    worker_id : string identifier e.g. "worker_01"

    Returns a dict:
    {
        "filtered_keypoints": [...],   # keypoints with low-conf ones zeroed out
        "freeze_state": {
            "upper": True/False,
            "core":  True/False,
            "lower": True/False
        },
        "frozen_zones": [...],         # list of zone names currently frozen
        "reason": "..."                # human-readable explanation
    }
    """

    if worker_id not in _freeze_store:
        _freeze_store[worker_id] = {
            "keypoints": keypoints,
            "freeze_state": {"upper": False, "core": False, "lower": False}
        }

    freeze_state = {"upper": False, "core": False, "lower": False}
    frozen_zones = []
    reason_parts = []

    filtered_keypoints = []
    for kp in keypoints:
        x, y, conf = kp[0], kp[1], kp[2]
        if conf >= CONFIDENCE_THRESHOLD:
            filtered_keypoints.append([x, y, conf])
        else:
            filtered_keypoints.append([0, 0, 0])   # zero out low-confidence keypoint

    # Check each zone
    for zone_name, indices in BODY_ZONES.items():
        low_conf_count = 0
        for idx in indices:
            if filtered_keypoints[idx][2] == 0:
                low_conf_count += 1
        ratio = low_conf_count / len(indices)
        if ratio > FREEZE_RATIO:
            freeze_state[zone_name] = True
            frozen_zones.append(zone_name)
            reason_parts.append(
                f"{zone_name} zone frozen ({low_conf_count}/{len(indices)} keypoints below threshold)"
            )

    # If zone is frozen, restore last known good keypoints for that zone
    last_good = _freeze_store[worker_id]["keypoints"]
    for zone_name, indices in BODY_ZONES.items():
        if freeze_state[zone_name]:
            for idx in indices:
                filtered_keypoints[idx] = last_good[idx]

    # Update freeze store only if NOT fully frozen (at least one zone is clean)
    if not all(freeze_state.values()):
        _freeze_store[worker_id]["keypoints"] = filtered_keypoints
        _freeze_store[worker_id]["freeze_state"] = freeze_state

    reason = "; ".join(reason_parts) if reason_parts else "All zones clear"

    return {
        "filtered_keypoints": filtered_keypoints,
        "freeze_state": freeze_state,
        "frozen_zones": frozen_zones,
        "reason": reason
    }


def reset_worker(worker_id):
    """Call this when a worker leaves the frame."""
    if worker_id in _freeze_store:
        del _freeze_store[worker_id]