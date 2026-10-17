

import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate angle at point B formed by points A-B-C.
    a, b, c : each is [x, y, conf]
    Returns angle in degrees (0-180).
    Returns None if any point has zero/low confidence.
    """
    if a[2] < 0.2 or b[2] < 0.2 or c[2] < 0.2:
        return None
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosine = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return round(np.degrees(np.arccos(cosine)), 1)


def score_trunk_flexion(angle):
    """
    Trunk bend: shoulder-hip-knee angle.
    Upright standing ≈ 160-170°. Forward bend → angle decreases.
    REBA: >60° forward flexion = high risk (angle < 110° here).
    """
    if angle is None:
        return 5   # unknown → mid risk
    if angle >= 155:
        return 1   # fully upright
    elif angle >= 140:
        return 3   # slight lean — acceptable
    elif angle >= 120:
        return 6   # AMBER — noticeable forward bend (≈20-40° flexion)
    elif angle >= 100:
        return 8   # RED zone — significant bend (≈40-60° flexion)
    else:
        return 10  # severe forward bend (>60° flexion) — immediate risk


def score_shoulder_elevation(angle):
    """
    Shoulder raise: hip-shoulder-elbow angle.
    Arms relaxed at side ≈ 10-30°. Raised overhead = higher.
    REBA: >90° arm elevation = high risk.
    """
    if angle is None:
        return 5
    if angle <= 25:
        return 1   # arms naturally at side
    elif angle <= 50:
        return 3   # slight raise — acceptable
    elif angle <= 75:
        return 6   # AMBER — arms raised notably
    elif angle <= 100:
        return 8   # RED — shoulder above mid-line
    else:
        return 10  # arms fully raised / overhead


def score_elbow_flexion(angle):
    """
    Elbow bend: shoulder-elbow-wrist angle.
    Comfortable working range: 70-120°.
    REBA: <60° or >135° = elevated risk.
    """
    if angle is None:
        return 5
    if 70 <= angle <= 120:
        return 1   # comfortable working range
    elif 55 <= angle < 70 or 120 < angle <= 140:
        return 4   # AMBER — slightly outside range
    elif 35 <= angle < 55 or 140 < angle <= 165:
        return 7   # RED — notably awkward
    else:
        return 9   # extreme: near full extension or full flexion


def score_knee_flexion(angle):
    """
    Knee bend: hip-knee-ankle angle.
    Standing straight ≈ 170°. Deep squat = lower angle.
    REBA: >60° knee flexion = high risk (angle < 120° here).
    """
    if angle is None:
        return 5
    if angle >= 160:
        return 1   # standing straight
    elif angle >= 140:
        return 3   # slight bend — acceptable
    elif angle >= 115:
        return 6   # AMBER — semi-squat
    elif angle >= 85:
        return 8   # RED — deep bend
    else:
        return 10  # full squat / kneeling


def compute_risk(keypoints):
    """
    keypoints : list/array of 17 items, each [x, y, conf]
    Returns dict with angles, joint_scores, composite_score, posture_color.
    """
    kp = keypoints

    # YOLO keypoint indices
    l_shoulder = kp[5];  r_shoulder = kp[6]
    l_elbow    = kp[7];  r_elbow    = kp[8]
    l_wrist    = kp[9];  r_wrist    = kp[10]
    l_hip      = kp[11]; r_hip      = kp[12]
    l_knee     = kp[13]; r_knee     = kp[14]
    l_ankle    = kp[15]; r_ankle    = kp[16]

    # ── Calculate angles ────────────────────────────────────────────────────
    trunk_l = calculate_angle(l_shoulder, l_hip, l_knee)
    trunk_r = calculate_angle(r_shoulder, r_hip, r_knee)
    trunk   = trunk_l if trunk_l is not None else trunk_r

    shoulder_l = calculate_angle(l_hip, l_shoulder, l_elbow)
    shoulder_r = calculate_angle(r_hip, r_shoulder, r_elbow)

    elbow_l = calculate_angle(l_shoulder, l_elbow, l_wrist)
    elbow_r = calculate_angle(r_shoulder, r_elbow, r_wrist)

    knee_l = calculate_angle(l_hip, l_knee, l_ankle)
    knee_r = calculate_angle(r_hip, r_knee, r_ankle)

    # ── Score each joint ────────────────────────────────────────────────────
    s_trunk  = score_trunk_flexion(trunk)
    s_shl_l  = score_shoulder_elevation(shoulder_l)
    s_shl_r  = score_shoulder_elevation(shoulder_r)
    s_elb_l  = score_elbow_flexion(elbow_l)
    s_elb_r  = score_elbow_flexion(elbow_r)
    s_knee_l = score_knee_flexion(knee_l)
    s_knee_r = score_knee_flexion(knee_r)

    # ── Weighted composite (REBA-inspired weights) ──────────────────────────
    # Trunk 40% — highest injury risk (spine / LBDs)
    # Shoulders 30% — common WMSD site in warehouse workers
    # Elbows 15%, Knees 15%
    composite = (
        s_trunk                      * 0.40 +
        max(s_shl_l, s_shl_r)        * 0.30 +
        max(s_elb_l, s_elb_r)        * 0.15 +
        max(s_knee_l, s_knee_r)      * 0.15
    )
    composite = round(composite, 2)

    # ── Colour coding ───────────────────────────────────────────────────────
    # Lowered thresholds so realistic warehouse postures register correctly
    if composite <= 3.0:
        color = "GREEN"
    elif composite <= 5.5:
        color = "AMBER"
    else:
        color = "RED"

    return {
        "angles": {
            "trunk":          trunk,
            "shoulder_left":  shoulder_l,
            "shoulder_right": shoulder_r,
            "elbow_left":     elbow_l,
            "elbow_right":    elbow_r,
            "knee_left":      knee_l,
            "knee_right":     knee_r,
            "trunk_flexion":  trunk,           # alias for POSTURE_KB lookup
            "shoulder_elevation": max(
                shoulder_l if shoulder_l else 0,
                shoulder_r if shoulder_r else 0),
            "elbow_flexion":  elbow_l if elbow_l else elbow_r,
            "knee_flexion":   knee_l  if knee_l  else knee_r,
        },
        "joint_scores": {
            "trunk":            s_trunk  / 10,
            "shoulder_left":    s_shl_l  / 10,
            "shoulder_right":   s_shl_r  / 10,
            "elbow_left":       s_elb_l  / 10,
            "elbow_right":      s_elb_r  / 10,
            "knee_left":        s_knee_l / 10,
            "knee_right":       s_knee_r / 10,
            # aliases for POSTURE_KB joint keys
            "trunk_flexion":        s_trunk  / 10,
            "shoulder_elevation":   max(s_shl_l, s_shl_r) / 10,
            "elbow_flexion":        max(s_elb_l, s_elb_r) / 10,
            "knee_flexion":         max(s_knee_l, s_knee_r) / 10,
        },
        "composite_score": composite,
        "posture_color":   color,
    }