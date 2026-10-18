# ergonomic_scorer.py
# Novelty 2 — Ergonomic Composite Risk Scorer
# Joint angle computation → 1-10 risk score → GREEN / AMBER / RED
# F3 Pipeline | Warehouse Intelligence System | J26-DS-352
#
# ANGLE GEOMETRY (YOLO + MediaPipe hybrid keypoints, screen coords):
#   Trunk    : shoulder-hip-knee      → upright ≈ 160-170°, bent forward = lower
#   Neck     : ear-neck-shoulder      → upright ≈ 160-170°, head forward = lower
#   Shoulder : hip-shoulder-elbow     → arms down ≈ 0-30°, raised = higher
#   Elbow    : shoulder-elbow-wrist   → comfortable 70-120°
#   Knee     : hip-knee-ankle/heel    → standing ≈ 170°, squat = lower
#   Wrist    : elbow-wrist-hand       → neutral ≈ 160-180°, deviated = lower
#
# REBA weights (Hignett & McAtamney 2000, Applied Ergonomics 31(2):201-205):
#   Trunk 35% | Neck 15% | Shoulder 25% | Elbow 10% | Knee 10% | Wrist 5%
#
# Hybrid scoring: if neck/wrist available (MediaPipe), use them.
#                 if missing (YOLO-only), return 2 (small neutral — not penalised).

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


# ── Individual joint scorers ──────────────────────────────────────────────────

def score_trunk_flexion(angle):
    """
    Trunk bend: shoulder-hip-knee angle.
    Upright standing ≈ 160-170°. Forward bend → angle decreases.
    REBA: >60° forward flexion = high risk (angle < 110° here).
    """
    if angle is None:
        return 2   # keypoint missing → small neutral contribution, not penalised
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


def score_neck_flexion(angle):
    """
    Neck bend: ear-neck-shoulder angle in screen coordinates.
    When standing upright the ear sits almost directly above the neck/shoulder,
    so the ear→neck→shoulder angle is SMALL (≈20-50°).
    As the head tilts forward the ear moves forward and the angle OPENS UP.
    So: small angle = upright (safe), large angle = forward flexion (risk).
    REBA: >20° neck flexion = risk; >45° = high risk.
    Only available in hybrid mode (MediaPipe neck landmark).
    """
    if angle is None:
        return 2   # neck not available (YOLO-only mode) — neutral contribution
    if angle <= 40:
        return 1   # head fully upright — ear directly above shoulder
    elif angle <= 65:
        return 2   # slight forward tilt — acceptable
    elif angle <= 90:
        return 5   # AMBER — moderate head-forward
    elif angle <= 120:
        return 7   # RED — significant forward head posture
    else:
        return 9   # severe neck flexion — immediate risk


def score_shoulder_elevation(angle):
    """
    Shoulder raise: hip-shoulder-elbow angle.
    Arms relaxed at side ≈ 10-30°. Raised overhead = higher.
    REBA: >90° arm elevation = high risk.
    """
    if angle is None:
        return 2   # keypoint missing → small neutral contribution, not penalised
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
    Arms hanging straight at side = ~170-180° → score 1 (neutral/resting).
    Comfortable working range: 70-120°.
    REBA risk: <60° (over-flexed) or sustained awkward extension.
    Near-straight (>140°) = arms at rest or reaching, not a risk by itself.
    """
    if angle is None:
        return 2   # keypoint missing → small neutral contribution, not penalised
    if angle > 140:
        return 1   # arms hanging or reaching straight — neutral resting posture
    elif 70 <= angle <= 140:
        return 1   # comfortable working range
    elif 55 <= angle < 70:
        return 4   # AMBER — slightly over-flexed
    elif 35 <= angle < 55:
        return 7   # RED — significantly over-flexed
    else:
        return 9   # extreme hyper-flexion


def score_knee_flexion(angle):
    """
    Knee bend: hip-knee-ankle angle.
    Standing straight ≈ 170°. Deep squat = lower angle.
    REBA: >60° knee flexion = high risk (angle < 120° here).
    """
    if angle is None:
        return 2   # keypoint missing → small neutral contribution, not penalised
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


def score_wrist_deviation(angle):
    """
    Wrist deviation: elbow-wrist-hand (finger base) angle.
    Neutral wrist ≈ 160-180°. Deviation increases risk.
    Only available in hybrid mode (MediaPipe hand landmarks).
    REBA: >15° deviation = risk.
    """
    if angle is None:
        return 2   # wrist not available (YOLO-only mode) — neutral contribution
    if angle >= 160:
        return 1   # neutral wrist
    elif angle >= 140:
        return 3   # slight deviation — acceptable
    elif angle >= 110:
        return 6   # AMBER — moderate deviation
    elif angle >= 80:
        return 8   # RED — significant deviation
    else:
        return 10  # extreme wrist bend


# ── Main risk computation ─────────────────────────────────────────────────────

def compute_risk(keypoints, unified=None):
    """
    keypoints : list of 17+ items, each [x, y, conf]  (YOLO format, index 0-16)
    unified   : dict of named joints from hybrid_pose (optional)
                If provided, neck and wrist scores are included.

    Returns dict with angles, joint_scores, composite_score, posture_color,
    and a 'mode' field ('hybrid' or 'yolo_only').
    """
    kp = keypoints

    # ── YOLO keypoint indices ────────────────────────────────────────────────
    l_shoulder = kp[5];  r_shoulder = kp[6]
    l_elbow    = kp[7];  r_elbow    = kp[8]
    l_wrist    = kp[9];  r_wrist    = kp[10]
    l_hip      = kp[11]; r_hip      = kp[12]
    l_knee     = kp[13]; r_knee     = kp[14]
    l_ankle    = kp[15]; r_ankle    = kp[16]

    # ── Calculate YOLO-based angles ──────────────────────────────────────────
    trunk_l = calculate_angle(l_shoulder, l_hip, l_knee)
    trunk_r = calculate_angle(r_shoulder, r_hip, r_knee)
    trunk   = trunk_l if trunk_l is not None else trunk_r

    shoulder_l = calculate_angle(l_hip, l_shoulder, l_elbow)
    shoulder_r = calculate_angle(r_hip, r_shoulder, r_elbow)

    elbow_l = calculate_angle(l_shoulder, l_elbow, l_wrist)
    elbow_r = calculate_angle(r_shoulder, r_elbow, r_wrist)

    knee_l = calculate_angle(l_hip, l_knee, l_ankle)
    knee_r = calculate_angle(r_hip, r_knee, r_ankle)

    # ── MediaPipe extra joints (only if unified dict provided) ───────────────
    neck_angle  = None
    wrist_l_dev = None
    wrist_r_dev = None
    mode        = "yolo_only"

    if unified is not None:
        mode = "hybrid"
        neck    = unified.get("neck",        [0, 0, 0])
        l_ear   = unified.get("left_ear",    [0, 0, 0])
        r_ear   = unified.get("right_ear",   [0, 0, 0])
        l_heel  = unified.get("left_heel",   [0, 0, 0])
        r_heel  = unified.get("right_heel",  [0, 0, 0])
        l_foot  = unified.get("left_foot",   [0, 0, 0])
        r_foot  = unified.get("right_foot",  [0, 0, 0])

        # Neck flexion: ear → neck → shoulder
        neck_l = calculate_angle(l_ear, neck, l_shoulder)
        neck_r = calculate_angle(r_ear, neck, r_shoulder)
        neck_angle = neck_l if neck_l is not None else neck_r

        # Use heel for knee angle if available (more accurate than ankle)
        knee_l_h = calculate_angle(l_hip, l_knee, l_heel) if l_heel[2] > 0.2 else knee_l
        knee_r_h = calculate_angle(r_hip, r_knee, r_heel) if r_heel[2] > 0.2 else knee_r
        if knee_l_h is not None: knee_l = knee_l_h
        if knee_r_h is not None: knee_r = knee_r_h

        # Wrist deviation: not reliably computable from pose-only landmarks
        # (would need hand landmark model for finger direction).
        # Leave as None → score_wrist_deviation returns 2 (neutral).
        wrist_l_dev = None
        wrist_r_dev = None

    # ── Score each joint ─────────────────────────────────────────────────────
    s_trunk   = score_trunk_flexion(trunk)
    s_neck    = score_neck_flexion(neck_angle)
    s_shl_l   = score_shoulder_elevation(shoulder_l)
    s_shl_r   = score_shoulder_elevation(shoulder_r)
    s_elb_l   = score_elbow_flexion(elbow_l)
    s_elb_r   = score_elbow_flexion(elbow_r)
    s_knee_l  = score_knee_flexion(knee_l)
    s_knee_r  = score_knee_flexion(knee_r)
    s_wrist_l = score_wrist_deviation(wrist_l_dev)
    s_wrist_r = score_wrist_deviation(wrist_r_dev)

    # ── Weighted composite ───────────────────────────────────────────────────
    # Weights: Trunk 40% | Shoulder 30% | Elbow 15% | Knee 15%
    # Neck excluded from composite — ear→neck→shoulder geometry in screen coords
    # is unreliable for crop images (partial body, varying camera angle).
    # Elbow near-straight (arms at rest) correctly scores 1, not penalised.
    composite = (
        s_trunk                        * 0.40 +
        max(s_shl_l, s_shl_r)          * 0.30 +
        max(s_elb_l, s_elb_r)          * 0.15 +
        max(s_knee_l, s_knee_r)        * 0.15
    )
    composite = round(composite, 2)

    # ── Colour coding ────────────────────────────────────────────────────────
    if composite <= 2.5:
        color = "GREEN"
    elif composite <= 4.0:
        color = "AMBER"
    else:
        color = "RED"

    return {
        "mode": mode,
        "angles": {
            "trunk":            trunk,
            "neck":             neck_angle,
            "shoulder_left":    shoulder_l,
            "shoulder_right":   shoulder_r,
            "elbow_left":       elbow_l,
            "elbow_right":      elbow_r,
            "knee_left":        knee_l,
            "knee_right":       knee_r,
            "wrist_left":       wrist_l_dev,
            "wrist_right":      wrist_r_dev,
            # aliases for dashboard lookup
            "trunk_flexion":    trunk,
            "neck_flexion":     neck_angle,
            "shoulder_elevation": max(
                shoulder_l if shoulder_l else 0,
                shoulder_r if shoulder_r else 0),
            "elbow_flexion":    elbow_l if elbow_l else elbow_r,
            "knee_flexion":     knee_l  if knee_l  else knee_r,
            "wrist_deviation":  max(
                wrist_l_dev if wrist_l_dev else 0,
                wrist_r_dev if wrist_r_dev else 0),
        },
        "joint_scores": {
            "trunk":            s_trunk   / 10,
            "neck":             s_neck    / 10,
            "shoulder_left":    s_shl_l   / 10,
            "shoulder_right":   s_shl_r   / 10,
            "elbow_left":       s_elb_l   / 10,
            "elbow_right":      s_elb_r   / 10,
            "knee_left":        s_knee_l  / 10,
            "knee_right":       s_knee_r  / 10,
            "wrist_left":       s_wrist_l / 10,
            "wrist_right":      s_wrist_r / 10,
            # aliases
            "trunk_flexion":        s_trunk             / 10,
            "neck_flexion":         s_neck              / 10,
            "shoulder_elevation":   max(s_shl_l, s_shl_r)   / 10,
            "elbow_flexion":        max(s_elb_l, s_elb_r)   / 10,
            "knee_flexion":         max(s_knee_l, s_knee_r) / 10,
            "wrist_deviation":      max(s_wrist_l, s_wrist_r) / 10,
        },
        "composite_score": composite,
        "posture_color":   color,
    }