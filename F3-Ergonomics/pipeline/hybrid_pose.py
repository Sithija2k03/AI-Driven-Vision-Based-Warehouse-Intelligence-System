

import cv2
import numpy as np

# ── MediaPipe landmark indices (from 33-point model) ─────────────────────────
# Full map: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP_NOSE          = 0
MP_LEFT_EYE      = 2
MP_RIGHT_EYE     = 5
MP_LEFT_EAR      = 7
MP_RIGHT_EAR     = 8
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER= 12
MP_LEFT_ELBOW    = 13
MP_RIGHT_ELBOW   = 14
MP_LEFT_WRIST    = 15
MP_RIGHT_WRIST   = 16
MP_LEFT_HIP      = 23
MP_RIGHT_HIP     = 24
MP_LEFT_KNEE     = 25
MP_RIGHT_KNEE    = 26
MP_LEFT_ANKLE    = 27
MP_RIGHT_ANKLE   = 28
MP_LEFT_HEEL     = 29
MP_RIGHT_HEEL    = 30
MP_LEFT_FOOT     = 31
MP_RIGHT_FOOT    = 32
# Neck is NOT a MediaPipe landmark — computed as midpoint of shoulders
# Left/Right pinky: 17/18, index: 19/20, thumb: 21/22

# ── YOLO→MediaPipe index mapping (for cross-validation) ──────────────────────
YOLO_TO_MP = {
    0:  MP_NOSE,
    5:  MP_LEFT_SHOULDER,   6:  MP_RIGHT_SHOULDER,
    7:  MP_LEFT_ELBOW,      8:  MP_RIGHT_ELBOW,
    9:  MP_LEFT_WRIST,      10: MP_RIGHT_WRIST,
    11: MP_LEFT_HIP,        12: MP_RIGHT_HIP,
    13: MP_LEFT_KNEE,       14: MP_RIGHT_KNEE,
    15: MP_LEFT_ANKLE,      16: MP_RIGHT_ANKLE,
}

# Confidence discount for MediaPipe-derived joints (visibility ≠ YOLO conf)
MP_CONF_SCALE = 0.85


def _load_mediapipe():
    """Lazy-load MediaPipe Pose — supports both legacy (0.9.x) and new (0.10+) API."""
    try:
        import mediapipe as mp

        # Try legacy solutions API first (mediapipe < 0.10)
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
            pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.3,
            )
            return pose, mp

        # New Tasks API (mediapipe >= 0.10)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, os, tempfile

        # Download the pose landmarker model if not cached
        model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_heavy.task")
        if not os.path.exists(model_path):
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                   "pose_landmarker_heavy.task")
            print("[hybrid_pose] Downloading MediaPipe pose model (~30 MB)…")
            urllib.request.urlretrieve(url, model_path)
            print("[hybrid_pose] Model downloaded.")

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        pose = mp_vision.PoseLandmarker.create_from_options(opts)
        # Tag it so run_hybrid knows which API to call
        pose._uses_tasks_api = True
        return pose, mp

    except Exception as e:
        print(f"[hybrid_pose] MediaPipe unavailable ({e}), using YOLO-only mode.")
        return None, None


# Module-level singleton so we don't reload per frame
_MP_POSE = None
_MP_LIB  = None


def _get_mediapipe():
    global _MP_POSE, _MP_LIB
    if _MP_POSE is None:
        _MP_POSE, _MP_LIB = _load_mediapipe()
    return _MP_POSE, _MP_LIB


def run_hybrid(img_path: str, yolo_model) -> dict:
    """
    Run YOLO + MediaPipe on one image, return unified keypoint dict.

    Returns
    -------
    dict with keys:
        yolo_kp   : list of 17 [x, y, conf]   — raw YOLO keypoints
        mp_kp     : list of 33 [x, y, conf]   — MediaPipe landmarks (or None)
        unified   : dict of named joints → [x, y, conf]
        det_conf  : float  — YOLO bounding-box confidence
        bbox      : [x1,y1,x2,y2]
        source    : "hybrid" | "yolo_only"
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]

    # ── 1. YOLO ──────────────────────────────────────────────────────────────
    results = yolo_model(img_path, verbose=False, conf=0.10)
    if not results or results[0].keypoints is None:
        return None
    kp_data = results[0].keypoints.data
    if len(kp_data) == 0:
        return None

    yolo_kp  = kp_data[0].cpu().numpy().tolist()   # 17 × [x, y, conf]
    det_conf = 0.0
    bbox     = None
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        box_data = results[0].boxes[0]
        bbox     = box_data.xyxy[0].cpu().numpy().tolist()
        det_conf = float(box_data.conf[0].cpu().numpy())

    if bbox is None:
        xs = [kp[0] for kp in yolo_kp if kp[2] > 0.1]
        ys = [kp[1] for kp in yolo_kp if kp[2] > 0.1]
        bbox = ([max(0,min(xs)-10), max(0,min(ys)-10),
                 max(xs)+10, max(ys)+10] if xs else [0,0,w,h])

    # ── 2. MediaPipe ─────────────────────────────────────────────────────────
    mp_pose, mp_lib = _get_mediapipe()
    mp_kp    = None
    mp_lm    = None
    source   = "yolo_only"

    if mp_pose is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        uses_tasks = getattr(mp_pose, '_uses_tasks_api', False)

        if uses_tasks:
            # New Tasks API (mediapipe >= 0.10)
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            mp_res   = mp_pose.detect(mp_image)
            if mp_res.pose_landmarks and len(mp_res.pose_landmarks) > 0:
                lms = mp_res.pose_landmarks[0]
                mp_kp = [
                    [lm.x * w, lm.y * h, lm.visibility * MP_CONF_SCALE]
                    for lm in lms
                ]
                mp_lm = lms
                source = "hybrid"
        else:
            # Legacy solutions API (mediapipe < 0.10)
            mp_res = mp_pose.process(img_rgb)
            if mp_res.pose_landmarks:
                lms = mp_res.pose_landmarks.landmark
                mp_kp = [
                    [lm.x * w, lm.y * h, lm.visibility * MP_CONF_SCALE]
                    for lm in lms
                ]
                mp_lm = lms
                source = "hybrid"

    # ── 3. Build unified joint dict ───────────────────────────────────────────
    # Start from YOLO 17 (authoritative for confidence filter)
    YOLO_NAMES = [
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    unified = {name: yolo_kp[i] for i, name in enumerate(YOLO_NAMES)}

    if mp_kp is not None:
        # Neck — midpoint of MP shoulders (not in YOLO at all)
        l_sh = mp_kp[MP_LEFT_SHOULDER]
        r_sh = mp_kp[MP_RIGHT_SHOULDER]
        neck_conf = min(l_sh[2], r_sh[2])
        unified["neck"] = [
            (l_sh[0] + r_sh[0]) / 2,
            (l_sh[1] + r_sh[1]) / 2 - (h * 0.05),  # slight upward offset
            neck_conf,
        ]

        # Heels (better ankle grounding for knee angle)
        unified["left_heel"]  = mp_kp[MP_LEFT_HEEL]
        unified["right_heel"] = mp_kp[MP_RIGHT_HEEL]

        # Foot index (toe tip — for stride/posture)
        unified["left_foot"]  = mp_kp[MP_LEFT_FOOT]
        unified["right_foot"] = mp_kp[MP_RIGHT_FOOT]

        # For YOLO joints with low confidence, cross-validate with MediaPipe
        # If YOLO conf < 0.25 but MP visibility > 0.5, prefer MediaPipe value
        for yolo_idx, mp_idx in YOLO_TO_MP.items():
            yname = YOLO_NAMES[yolo_idx]
            y_kp  = yolo_kp[yolo_idx]
            m_kp  = mp_kp[mp_idx]
            if y_kp[2] < 0.25 and m_kp[2] > 0.50:
                # Replace low-conf YOLO with MediaPipe, but mark as estimated
                unified[yname] = [m_kp[0], m_kp[1], m_kp[2] * 0.80]

    return {
        "yolo_kp":  yolo_kp,
        "mp_kp":    mp_kp,
        "unified":  unified,
        "det_conf": det_conf,
        "bbox":     bbox,
        "source":   source,
        "img_wh":   (w, h),
    }


def unified_to_yolo_list(unified: dict) -> list:
    """
    Convert unified dict back to 17-item YOLO-format list for
    compatibility with confidence_filter and ergonomic_scorer (legacy path).
    Extra joints (neck, heel, foot) are appended AFTER index 16.
    """
    YOLO_NAMES = [
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    base = [unified.get(n, [0, 0, 0]) for n in YOLO_NAMES]

    # Append extras so ergonomic_scorer can use indices 17+
    extras = ["neck", "left_heel", "right_heel", "left_foot", "right_foot"]
    for name in extras:
        base.append(unified.get(name, [0, 0, 0]))

    return base   # 17 + 5 = 22 items; scorer uses named access via unified dict