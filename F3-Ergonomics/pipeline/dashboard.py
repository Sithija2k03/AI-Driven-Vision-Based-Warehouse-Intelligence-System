# dashboard.py
# F3 Pipeline — Ergonomic Risk Dashboard (Streamlit) v2
# Warehouse Intelligence System | J26-DS-352 | SLIIT
#
# Run: python -m streamlit run D:\F3-Project\pipeline\dashboard.py

import sys
import io
import base64
import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Path setup ────────────────────────────────────────────────────────────────
PIPELINE_DIR  = r"D:\F3-Project\pipeline"
SOURCE_DIRS   = [
    r"D:\F3-Project\data\worker-crops",   # highest priority — per-person crops
    r"D:\F3-Project\data\crops",          # additional per-person crops
    # frames/ excluded: contains full-scene CCTV frames, not individual person crops
]
YOLO_MODEL    = r"D:\F3-Project\models\yolo11m-pose.pt"

sys.path.insert(0, PIPELINE_DIR)

from confidence_filter   import check_confidence, reset_worker
from ergonomic_scorer    import compute_risk
from occlusion_handler   import apply_occlusion_handling, save_report
from predict_demographic import load_demographic_model, predict_demographic
from hybrid_pose         import run_hybrid

OCCLUSION_REPORT_DIR = r"D:\F3-Project\pipeline\occlusion_reports"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F3 Ergonomic Risk Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0A0F2C !important; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1A2150; border: 1px solid #2D3A7A;
    color: white !important; border-radius: 8px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1E90FF; border-color: #1E90FF;
}

/* Main */
.main .block-container {
    background-color: #F0F4FF;
    padding-top: 1.2rem;
    max-width: 100%;
}

/* ── Stat tile ── */
.stile {
    background:#FFFFFF; border-radius:12px;
    padding:16px 10px; box-shadow:0 2px 8px rgba(0,0,0,0.06);
    text-align:center; border-top: 3px solid #1E90FF;
}
.stile-val { font-size:1.9rem; font-weight:700; color:#0A0F2C; }
.stile-lbl { font-size:0.72rem; color:#718096; margin-top:2px; font-weight:600; letter-spacing:0.03em; }

/* ── Uniform worker card ── */
.wcard-outer {
    background: #FFFFFF;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 3px 12px rgba(0,0,0,0.10);
    border: 3px solid #ccc;
    width: 100%;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
}
.wcard-outer:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }
.wcard-green { border-color: #00C853 !important; }
.wcard-amber { border-color: #FFB300 !important; }
.wcard-red   { border-color: #FF3D00 !important; }

/* confidence bar at top of card */
.conf-bar {
    width: 100%; padding: 4px 8px;
    font-size: 0.7rem; font-weight: 700;
    color: white; letter-spacing: 0.04em;
    display: flex; justify-content: space-between;
}
.conf-green { background: #00C853; }
.conf-amber { background: #FFB300; }
.conf-red   { background: #FF3D00; }

/* image wrapper — fixed square */
.wcard-img-wrap {
    width: 100%; aspect-ratio: 1/1;
    overflow: hidden; background: #111;
    display: flex; align-items: center; justify-content: center;
}
.wcard-img-wrap img { width:100%; height:100%; object-fit:cover; }

/* card footer */
.wcard-footer {
    padding: 8px 10px 10px;
    background: #FFFFFF;
    text-align: center;
}
.wcard-id   { font-size:0.82rem; font-weight:700; color:#0A0F2C; }
.wcard-score { font-size:1.25rem; font-weight:700; margin-top:2px; }
.wcard-trunk { font-size:0.68rem; color:#718096; }

/* Badge */
.badge {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:0.68rem; font-weight:700; letter-spacing:0.04em; margin-top:3px;
}
.badge-green { background:#E8F5E9; color:#00A040; }
.badge-amber { background:#FFF8E1; color:#B45309; }
.badge-red   { background:#FBE9E7; color:#CC2200; }

/* Section header */
.sec-hdr {
    font-size:1rem; font-weight:700; color:#0A0F2C;
    border-bottom:2px solid #CBD5E0; padding-bottom:6px; margin-bottom:14px;
}

/* Advisory */
.adv-green { background:#E8F5E9; border-left:4px solid #00C853; padding:12px 16px; border-radius:8px; color:#1B5E20; margin-top:10px; font-size:0.85rem; }
.adv-amber { background:#FFF8E1; border-left:4px solid #FFB300; padding:12px 16px; border-radius:8px; color:#7B4A00; margin-top:10px; font-size:0.85rem; }
.adv-red   { background:#FBE9E7; border-left:4px solid #FF3D00; padding:12px 16px; border-radius:8px; color:#BF360C; margin-top:10px; font-size:0.85rem; }

/* Zone tag */
.ztag { display:inline-block; background:#EEF2FF; color:#3730A3; padding:2px 10px; border-radius:20px; font-size:0.73rem; margin:2px; font-weight:600; }

/* Detail panel box */
.detail-panel {
    background: #FFFFFF; border-radius:14px;
    padding: 20px; box-shadow: 0 3px 16px rgba(0,0,0,0.08);
    border-top: 4px solid #1E90FF;
}

/* Demo pill */
.demo-pill { display:inline-block; background:#EEF2FF; color:#3730A3; padding:4px 14px; border-radius:20px; font-size:0.8rem; font-weight:600; margin-right:6px; }
</style>
""", unsafe_allow_html=True)


# ── YOLO loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_MODEL)


# ── Discover unique workers across all source dirs ────────────────────────────
def discover_workers():
    """
    Returns dict: { worker_id: [img_path, img_path, ...] }
    Scans ALL source dirs and MERGES frames for workers that appear in multiple
    sources — so a worker in both worker-crops AND frames contributes all their
    frames to the pool. Worst-frame selection then picks the highest-risk one.
    Duplicate paths (same file in two sources) are deduplicated by filename.
    """
    worker_map = {}   # { wid: { filename: full_path } }

    for src in SOURCE_DIRS:
        base = Path(src)
        if not base.exists():
            continue
        for folder in sorted(base.iterdir()):
            if not folder.is_dir():
                continue
            wid  = folder.name
            imgs = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
            if not imgs:
                continue
            if wid not in worker_map:
                worker_map[wid] = {}
            for p in imgs:
                # Key by filename so duplicate frames don't run twice
                worker_map[wid][p.name] = str(p)

    # Flatten to sorted list of paths per worker
    def sort_key(k):
        parts = k.replace("Worker-", "")
        return int(parts) if parts.isdigit() else 999

    return {
        wid: sorted(paths.values())
        for wid, paths in sorted(worker_map.items(), key=lambda x: sort_key(x[0]))
    }


# ── Score all frames for one worker, return worst-risk result ─────────────────
def score_all_frames(worker_id: str, img_paths: list):
    """
    Runs run_pipeline() on every frame for a worker.
    Returns the result dict with the highest composite risk score.
    Falls back to the first successful result if scores are tied or all zero.
    Also attaches 'all_frame_scores' list for audit / detail panel.
    """
    best_result  = None
    best_score   = -1.0
    all_scores   = []   # [(img_path, composite, color)]

    for img_path in img_paths:
        try:
            r = run_pipeline(img_path, worker_id)
            if r is None:
                all_scores.append((img_path, None, "ERROR"))
                continue
            score = r.get("composite", 0.0)
            all_scores.append((img_path, round(score, 2), r.get("color", "—")))
            if score > best_score:
                best_score  = score
                best_result = r
        except Exception as e:
            all_scores.append((img_path, None, f"ERR: {e}"))

    if best_result is not None:
        best_result["all_frame_scores"] = all_scores   # attach audit trail
        best_result["frame_count"]      = len(img_paths)
    return best_result


# ── Run pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(img_path: str, worker_id: str):
    """
    Hybrid YOLO + MediaPipe pipeline (Novelty 1 → 2 → 3).

    1. run_hybrid()       — YOLO 17kp + MediaPipe 33kp fused into unified dict
    2. confidence_filter  — Novelty 1: freeze low-confidence joints
    3. occlusion_handler  — bilateral symmetry mirroring
    4. compute_risk()     — Novelty 2: REBA-weighted ergonomic scoring (17+33kp)
    5. predict_demographic — Novelty 3: age/gender from crop
    """
    model   = load_yolo()
    hybrid  = run_hybrid(img_path, model)

    if hybrid is None:
        return None   # No person detected by either model

    keypoints = hybrid["yolo_kp"]    # 17-item YOLO list → confidence_filter
    unified   = hybrid["unified"]    # named dict including neck, heel, foot
    det_conf  = hybrid["det_conf"]
    bbox      = hybrid["bbox"]
    pose_src  = hybrid["source"]     # "hybrid" | "yolo_only"

    # ── Novelty 1: Confidence filter (operates on YOLO keypoints) ────────────
    reset_worker(worker_id)
    cf = check_confidence(keypoints, worker_id)
    filtered_kp  = cf["filtered_keypoints"]
    frozen_zones = cf["frozen_zones"]
    reason       = cf["reason"]

    # ── Occlusion Handling (delegated to occlusion_handler.py) ───────────────
    # Estimates missing keypoints via body-symmetry mirroring and writes a
    # per-worker JSON proof file to OCCLUSION_REPORT_DIR.
    img_w = Image.open(img_path).width
    fkp, estimated, occ_report = apply_occlusion_handling(
        keypoints   = filtered_kp,
        img_width   = img_w,
        worker_id   = worker_id,
        frame_label = Path(img_path).stem,
    )
    # Persist the proof artefact (non-blocking — skip silently if dir unwritable)
    try:
        saved_path = save_report(occ_report, out_dir=OCCLUSION_REPORT_DIR)
    except Exception:
        saved_path = None

    # ── Novelty 2: Ergonomic scoring — pass unified dict for neck + wrist ────
    # unified supplies neck, heel, foot joints from MediaPipe (missing in YOLO)
    risk = compute_risk(fkp, unified=unified)

    # ── Novelty 3: Demographic prediction ────────────────────────────────────
    demo_model, demo_meta = load_demographic_model()
    demo = predict_demographic(demo_model, demo_meta, img_path)

    return {
        "worker_id":     worker_id,
        "img_path":      img_path,
        "bbox":          bbox,
        "det_conf":      det_conf,
        "pose_source":   pose_src,   # "hybrid" = 17+33kp | "yolo_only" = 17kp
        "pose_mode":     risk.get("mode", pose_src),
        "angles":        risk["angles"],
        "joint_scores":  risk["joint_scores"],
        "composite":     risk["composite_score"],
        "color":         risk["posture_color"],
        "frozen_zones":  frozen_zones,
        "filter_reason": reason,
        "estimated_kp":  estimated,
        "occ_report":    occ_report,
        "occ_report_path": saved_path,
        "yolo_kp":       fkp,        # 17-item YOLO keypoints (post-filter+occlusion)
        "demo_age":      demo["age"],
        "demo_gender":   demo["gender"],
        "demo_age_conf": demo["age_conf"],
        "demo_gen_conf": demo["gender_conf"],
        "demo_status":   demo["status"],
    }


# ── Advisory text ─────────────────────────────────────────────────────────────
def get_advisory(result):
    color  = result["color"]
    angles = result.get("angles", {})
    trunk  = angles.get("trunk") or 180
    sh     = angles.get("shoulder_elevation") or 0
    knee   = angles.get("knee_flexion") or 180

    if color == "GREEN":
        return "green", "✅ Safe posture detected. No immediate action required."
    elif color == "AMBER":
        issues = []
        if trunk < 140: issues.append("moderate trunk flexion")
        if sh > 50:     issues.append("elevated arms")
        if knee < 140:  issues.append("knee bend")
        detail = ", ".join(issues) if issues else "posture outside comfortable range"
        return "amber", f"⚠️ Moderate risk — {detail}. Consider task rotation."
    else:
        issues = []
        if trunk < 120: issues.append("severe trunk flexion")
        if sh > 75:     issues.append("arms raised overhead")
        if knee < 115:  issues.append("deep knee bend")
        detail = ", ".join(issues) if issues else "multiple high-risk postures"
        return "red", f"🔴 High risk — {detail}. Rest break or reassignment recommended."


COLOR_HEX   = {"GREEN": "#00C853", "AMBER": "#FFB300", "RED": "#FF3D00"}
COLOR_RGB   = {"GREEN": (0, 200, 83), "AMBER": (255, 179, 0), "RED": (255, 61, 0)}


# ── Convert PIL image → base64 data URI (for HTML embedding) ─────────────────
def img_to_b64(img: Image.Image, size: int = 300) -> str:
    """Square-crop + resize to fixed size, return as base64 PNG data URI."""
    img = img.convert("RGB")
    w, h = img.size
    # Centre-crop to square
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    img  = img.crop((left, top, left + side, top + side))
    img  = img.resize((size, size), Image.LANCZOS)
    buf  = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
CARD_CLASS  = {"GREEN": "wcard-green", "AMBER": "wcard-amber", "RED": "wcard-red"}
BADGE_CLASS = {"GREEN": "badge-green", "AMBER": "badge-amber", "RED": "badge-red"}
ADV_CLASS   = {"green": "adv-green",   "amber": "adv-amber",   "red": "adv-red"}


# ── YOLO skeleton connections ─────────────────────────────────────────────────
SKELETON_PAIRS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6),   # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 11), (6, 12),  # torso sides
    (11, 12),          # hips
    # Lower body
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Joint index → joint name (for per-joint color in detail panel)
JOINT_SCORE_MAP = {
    # trunk joints: shoulder, hip, knee (avg trunk score)
    5: "shoulder_elevation", 6: "shoulder_elevation",
    7: "elbow_flexion",      8: "elbow_flexion",
    9: "wrist_deviation",    10: "wrist_deviation",
    11: "trunk_flexion",     12: "trunk_flexion",
    13: "knee_flexion",      14: "knee_flexion",
    15: "knee_flexion",      16: "knee_flexion",
}


def _score_to_rgb(score_01: float):
    """Map 0-1 joint score to RGB: green → amber → red."""
    s = float(score_01)
    if s <= 0.25:
        return (0, 200, 83)     # green
    elif s <= 0.45:
        return (255, 179, 0)    # amber
    else:
        return (255, 61, 0)     # red


def draw_skeleton(img: Image.Image, keypoints: list, joint_scores: dict = None,
                  overall_color: str = "GREEN", line_width: int = 3,
                  dot_radius: int = 5) -> Image.Image:
    """
    Draw skeleton overlay on a PIL image.

    keypoints   : 17-item list of [x, y, conf]
    joint_scores: dict of joint_name -> 0-1 score (for per-joint coloring)
                  If None, uses overall_color for all joints/bones.
    overall_color: 'GREEN'/'AMBER'/'RED' used when joint_scores is None
    """
    img = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    overall_rgb = COLOR_RGB.get(overall_color, (0, 200, 83))

    def kp_ok(idx):
        return idx < len(keypoints) and keypoints[idx][2] > 0.2

    def kp_xy(idx):
        return (int(keypoints[idx][0]), int(keypoints[idx][1]))

    def joint_rgb(idx):
        if joint_scores is None:
            return overall_rgb
        jname = JOINT_SCORE_MAP.get(idx)
        if jname and jname in joint_scores:
            return _score_to_rgb(joint_scores[jname])
        return overall_rgb

    # ── Draw limb lines ───────────────────────────────────────────────────────
    for a_idx, b_idx in SKELETON_PAIRS:
        if kp_ok(a_idx) and kp_ok(b_idx):
            ax, ay = kp_xy(a_idx)
            bx, by = kp_xy(b_idx)
            # Use midpoint joint color for the bone
            if joint_scores is not None:
                rgb = _score_to_rgb(
                    (joint_scores.get(JOINT_SCORE_MAP.get(a_idx, ""), 0.1) +
                     joint_scores.get(JOINT_SCORE_MAP.get(b_idx, ""), 0.1)) / 2
                )
            else:
                rgb = overall_rgb
            r, g, b = rgb
            draw.line([(ax, ay), (bx, by)], fill=(r, g, b, 210), width=line_width)

    # ── Draw joint dots ───────────────────────────────────────────────────────
    for idx in range(min(17, len(keypoints))):
        if kp_ok(idx):
            x, y = kp_xy(idx)
            r, g, b = joint_rgb(idx)
            # Outer white ring
            draw.ellipse(
                [(x - dot_radius - 2, y - dot_radius - 2),
                 (x + dot_radius + 2, y + dot_radius + 2)],
                fill=(255, 255, 255, 200)
            )
            # Colored fill
            draw.ellipse(
                [(x - dot_radius, y - dot_radius),
                 (x + dot_radius, y + dot_radius)],
                fill=(r, g, b, 230)
            )

    img = Image.alpha_composite(img, overlay).convert("RGB")
    return img


# ── Annotate frame with bounding box + confidence label ───────────────────────
def annotate_frame(img_path: str, bbox, det_conf: float, risk_color: str,
                   composite_score: float) -> Image.Image:
    """
    Draw coloured bounding box over the worker.
    Label above box: risk colour + detection confidence + composite score.
    bbox: [x1, y1, x2, y2] in pixels (from YOLO detection)
    det_conf: YOLO detection confidence 0.0–1.0
    risk_color: 'GREEN' / 'AMBER' / 'RED'
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    rgb = COLOR_RGB[risk_color]
    hex_color = COLOR_HEX[risk_color]

    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Box thickness scales with image size
    thickness = max(3, int(img.width / 120))

    # Draw bounding box
    for t in range(thickness):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=rgb)

    # Label text
    label = f"{risk_color}  conf:{det_conf:.2f}  score:{composite_score:.1f}/10"

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(img.width / 30)))
    except Exception:
        font = ImageFont.load_default()

    # Measure text size
    try:
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
    except Exception:
        text_w, text_h = len(label) * 8, 16

    # Background pill behind label
    pad = 4
    label_x = x1
    label_y = max(0, y1 - text_h - pad * 2 - thickness)
    draw.rectangle(
        [label_x, label_y, label_x + text_w + pad * 2, label_y + text_h + pad * 2],
        fill=rgb
    )

    # Label text in white
    draw.text((label_x + pad, label_y + pad), label, fill=(255, 255, 255), font=font)

    # Corner accent squares for clean look
    sq = thickness + 4
    corners = [(x1, y1), (x2 - sq, y1), (x1, y2 - sq), (x2 - sq, y2 - sq)]
    for cx, cy in corners:
        draw.rectangle([cx, cy, cx + sq, cy + sq], fill=rgb)

    return img


# ── Load all workers — worst-frame selection ──────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_workers():
    """
    For each worker, scores ALL frames and selects the one with the
    highest composite risk score (worst posture observed).
    Falls back gracefully for workers with only 1 frame or failed detections.
    """
    wmap    = discover_workers()
    results = []

    progress = st.progress(0, text="Scoring all frames…")
    total    = len(wmap)

    for i, (wid, img_paths) in enumerate(wmap.items()):
        progress.progress(
            (i + 1) / total,
            text=f"Scoring {wid} ({len(img_paths)} frame{'s' if len(img_paths)>1 else ''})…"
        )
        try:
            r = score_all_frames(wid, img_paths)
            if r:
                results.append(r)
            else:
                # All frames failed detection — show as GREEN/0.0 (undetected ≠ risky)
                results.append({
                    "worker_id":       wid,
                    "img_path":        img_paths[0],
                    "color":           "GREEN",
                    "composite":       0.0,
                    "det_conf":        0.0,
                    "angles":          {},
                    "joint_scores":    {},
                    "frozen_zones":    [],
                    "filter_reason":   "No pose detected in any frame",
                    "frame_count":     len(img_paths),
                    "all_frame_scores": [(p, None, "ERROR") for p in img_paths],
                    "error":           "YOLO found no person across all frames",
                    "demo_age":        "N/A",
                    "demo_gender":     "N/A",
                    "demo_age_conf":   0.0,
                    "demo_gen_conf":   0.0,
                    "demo_status":     "no detection",
                })
        except Exception as e:
            results.append({
                "worker_id":       wid,
                "img_path":        img_paths[0],
                "color":           "GREEN",
                "composite":       0.0,
                "det_conf":        0.0,
                "angles":          {},
                "joint_scores":    {},
                "frozen_zones":    [],
                "filter_reason":   "Pipeline error",
                "frame_count":     len(img_paths),
                "all_frame_scores": [],
                "error":           str(e),
                "demo_age":        "N/A",
                "demo_gender":     "N/A",
                "demo_age_conf":   0.0,
                "demo_gen_conf":   0.0,
                "demo_status":     "no detection",
            })

    progress.empty()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 F3 Pipeline")
    st.markdown("**Warehouse Intelligence System**")
    st.markdown("*J26-DS-352 · SLIIT*")
    st.markdown("---")

    st.markdown("### 📄 Pages")
    page = st.radio("", ["📊 Overview", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")

    st.markdown("### 🎯 Risk Filter")
    st.markdown("<small style='color:#A0AEC0'>Click to show workers by risk level</small>", unsafe_allow_html=True)

    # Risk filter buttons
    if "active_filter" not in st.session_state:
        st.session_state.active_filter = "ALL"

    col_g, col_a, col_r = st.columns(3)
    with col_g:
        if st.button("🟢 GREEN", use_container_width=True):
            st.session_state.active_filter = "GREEN"
    with col_a:
        if st.button("🟡 AMBER", use_container_width=True):
            st.session_state.active_filter = "AMBER"
    with col_r:
        if st.button("🔴 RED", use_container_width=True):
            st.session_state.active_filter = "RED"

    if st.button("👁 Show All", use_container_width=True):
        st.session_state.active_filter = "ALL"

    active = st.session_state.active_filter
    st.markdown(f"<small style='color:#A0AEC0'>Active: <b style='color:white'>{active}</b></small>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Pipeline", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <small style='color:#4A5568'>
    YOLOv11m-pose<br>
    Conf threshold: > 0.30<br>
    Scorer: REBA-weighted<br>
    Sources: worker-crops → crops → frames
    </small>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":

    st.markdown("## 📊 Worker Ergonomic Risk Overview")
    st.markdown("Real-time posture risk assessment · All sources combined")

    with st.spinner("Running F3 pipeline across all worker sources..."):
        all_results = load_all_workers()

    if not all_results:
        st.error("No workers found. Check that SOURCE_DIRS paths exist and contain Worker-N subfolders.")
        st.stop()

    # Apply filter
    filt = st.session_state.active_filter
    shown = all_results if filt == "ALL" else [r for r in all_results if r["color"] == filt]

    # ── Stat tiles ────────────────────────────────────────────────────────────
    total   = len(all_results)
    n_green = sum(1 for r in all_results if r["color"] == "GREEN")
    n_amber = sum(1 for r in all_results if r["color"] == "AMBER")
    n_red   = sum(1 for r in all_results if r["color"] == "RED")
    avg_s   = round(np.mean([r["composite"] for r in all_results]), 1) if all_results else 0

    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.markdown(f'<div class="stile"><div class="stile-val">{total}</div><div class="stile-lbl">Total Workers</div></div>', unsafe_allow_html=True)
    with t2:
        st.markdown(f'<div class="stile" style="border-top:3px solid #00C853"><div class="stile-val" style="color:#00C853">{n_green}</div><div class="stile-lbl">GREEN — Safe</div></div>', unsafe_allow_html=True)
    with t3:
        st.markdown(f'<div class="stile" style="border-top:3px solid #FFB300"><div class="stile-val" style="color:#FFB300">{n_amber}</div><div class="stile-lbl">AMBER — Monitor</div></div>', unsafe_allow_html=True)
    with t4:
        st.markdown(f'<div class="stile" style="border-top:3px solid #FF3D00"><div class="stile-val" style="color:#FF3D00">{n_red}</div><div class="stile-lbl">RED — Intervene</div></div>', unsafe_allow_html=True)
    with t5:
        st.markdown(f'<div class="stile" style="border-top:3px solid #1E90FF"><div class="stile-val" style="color:#1E90FF">{avg_s}</div><div class="stile-lbl">Avg Score /10</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not shown:
        st.info(f"No workers with {filt} risk level found.")
        st.stop()

    # ── Grid + Detail ──────────────────────────────────────────────────────────
    if "selected_worker" not in st.session_state:
        st.session_state.selected_worker = shown[0]["worker_id"]

    grid_col, detail_col = st.columns([3, 1.4], gap="large")

    # ── Worker grid — 4 per row, uniform square cards ─────────────────────────
    with grid_col:
        filt_label = filt if filt != "ALL" else "All Risk Levels"
        st.markdown(f'<div class="sec-hdr">👷 Workers — {filt_label} ({len(shown)})</div>',
                    unsafe_allow_html=True)

        COLS = 4
        rows = [shown[i:i+COLS] for i in range(0, len(shown), COLS)]

        for row in rows:
            cols = st.columns(COLS, gap="small")
            for col, result in zip(cols, row):
                with col:
                    color     = result["color"]
                    wid       = result["worker_id"]
                    score     = result.get("composite", 0)
                    det_conf  = result.get("det_conf", 0.0)
                    angles    = result.get("angles", {})
                    trunk     = angles.get("trunk")
                    trunk_str = f"{trunk}°" if trunk else "—"
                    ch        = COLOR_HEX[color]
                    cc        = f"wcard-{color.lower()}"
                    bc        = f"badge-{color.lower()}"
                    cconf     = f"conf-{color.lower()}"

                    # Build annotated image → skeleton overlay → square → base64
                    b64 = ""
                    try:
                        ann = annotate_frame(
                            result["img_path"],
                            result.get("bbox", [0, 0, 50, 50]),
                            det_conf, color, score,
                        )
                        # Grid card: skeleton in overall risk color (no per-joint split)
                        kp = result.get("yolo_kp")
                        if kp:
                            ann = draw_skeleton(ann, kp, overall_color=color)
                        b64 = img_to_b64(ann, size=280)
                    except Exception:
                        try:
                            b64 = img_to_b64(Image.open(result["img_path"]), size=280)
                        except Exception:
                            pass

                    img_tag = (f'<img src="{b64}" alt="{wid}">'
                               if b64 else
                               '<div style="color:#999;font-size:2rem;line-height:280px">🖼️</div>')

                    # Single HTML block = guaranteed uniform size
                    st.markdown(f"""
                    <div class="wcard-outer {cc}">
                        <div class="{cconf} conf-bar">
                            <span>{color}</span>
                            <span>conf {det_conf:.2f}</span>
                        </div>
                        <div class="wcard-img-wrap">{img_tag}</div>
                        <div class="wcard-footer">
                            <div class="wcard-id">{wid}</div>
                            <div class="wcard-score" style="color:{ch}">{score}<span style="font-size:0.7rem;color:#718096">/10</span></div>
                            <div class="wcard-trunk">Trunk: {trunk_str}</div>
                            <span class="badge {bc}">{color}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Invisible button triggers selection
                    if st.button("View detail →", key=f"btn_{wid}",
                                 use_container_width=True):
                        st.session_state.selected_worker = wid
                        st.rerun()

    # ── Detail panel ──────────────────────────────────────────────────────────
    with detail_col:
        st.markdown('<div class="sec-hdr">🔍 Worker Detail</div>', unsafe_allow_html=True)

        sel_id = st.session_state.selected_worker
        sel    = next((r for r in all_results if r["worker_id"] == sel_id), shown[0])

        color    = sel["color"]
        score    = sel.get("composite", 0)
        det_conf = sel.get("det_conf", 0.0)
        angles   = sel.get("angles", {})
        joints   = sel.get("joint_scores", {})
        frozen   = sel.get("frozen_zones", [])
        error    = sel.get("error")
        ch       = COLOR_HEX[color]
        bc       = f"badge-{color.lower()}"

        st.markdown(f'<div class="detail-panel">', unsafe_allow_html=True)

        # Annotated image — full width, with per-joint skeleton overlay
        try:
            ann_img = annotate_frame(
                sel["img_path"],
                sel.get("bbox", [0, 0, 100, 100]),
                det_conf, color, score,
            )
            # Detail panel: per-joint colors — each joint colored by its own score
            kp_sel = sel.get("yolo_kp")
            if kp_sel:
                ann_img = draw_skeleton(
                    ann_img, kp_sel,
                    joint_scores=sel.get("joint_scores"),
                    overall_color=color,
                    line_width=4,
                    dot_radius=7,
                )
            st.image(ann_img, use_container_width=True)
        except Exception:
            try:
                st.image(Image.open(sel["img_path"]), use_container_width=True)
            except Exception:
                pass

        # Header row
        hc1, hc2 = st.columns([1.5, 1])
        with hc1:
            st.markdown(f"### {sel_id}")
            st.markdown(f'<span class="badge {bc}" style="font-size:0.85rem;padding:4px 14px">{color}</span>', unsafe_allow_html=True)
        with hc2:
            st.metric("Score", f"{score}/10")
            st.markdown(f"<small style='color:#718096'>Conf: {det_conf:.2f}</small>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Demographic Profile (Novelty 3) ──────────────────────────────────
        st.markdown("**👤 Demographic Profile**")
        demo_age      = sel.get("demo_age",      "Pending")
        demo_gender   = sel.get("demo_gender",   "Pending")
        demo_age_conf = sel.get("demo_age_conf", 0.0)
        demo_gen_conf = sel.get("demo_gen_conf", 0.0)
        demo_status   = sel.get("demo_status",   "model not trained yet")

        if demo_status == "ok":
            # Confidence colour: green ≥ 0.75, amber ≥ 0.55, red < 0.55
            def conf_color(c):
                if c >= 0.75: return "#00C853"
                if c >= 0.55: return "#FFB300"
                return "#FF3D00"
            st.markdown(f"""
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:6px;">
                <div style="background:#EEF2FF;border-radius:10px;padding:10px 16px;
                            flex:1;min-width:120px;text-align:center;">
                    <div style="font-size:0.7rem;color:#718096;font-weight:600;
                                letter-spacing:0.05em;">AGE BAND</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#0A0F2C;">
                        {demo_age}
                    </div>
                    <div style="font-size:0.72rem;color:{conf_color(demo_age_conf)};">
                        conf {demo_age_conf:.2f}
                    </div>
                </div>
                <div style="background:#EEF2FF;border-radius:10px;padding:10px 16px;
                            flex:1;min-width:120px;text-align:center;">
                    <div style="font-size:0.7rem;color:#718096;font-weight:600;
                                letter-spacing:0.05em;">GENDER</div>
                    <div style="font-size:1.3rem;font-weight:700;color:#0A0F2C;">
                        {"♂ Male" if demo_gender=="Male" else "♀ Female"}
                    </div>
                    <div style="font-size:0.72rem;color:{conf_color(demo_gen_conf)};">
                        conf {demo_gen_conf:.2f}
                    </div>
                </div>
            </div>
            <small style='color:#A0AEC0'>
                ResNet-18 + MLP · Novelty 3
                {"· ⚠️ Gender confidence low — only 8 Female training samples" if demo_gender=="Female" and demo_gen_conf < 0.70 else ""}
            </small>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <span class="demo-pill">Age: {demo_age}</span>
            <span class="demo-pill">Gender: {demo_gender}</span>
            <br><small style='color:#A0AEC0'>Novelty 3 — {demo_status}</small>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if error:
            st.warning(f"⚠️ {error}")
        else:
            # ── Pose source badge ─────────────────────────────────────────────
            pose_src = sel.get("pose_source", "yolo_only")
            pose_mode = sel.get("pose_mode", pose_src)
            if pose_mode == "hybrid":
                st.markdown(
                    "<span style='background:#6C63FF;color:white;border-radius:6px;"
                    "padding:2px 10px;font-size:0.72rem;font-weight:700;letter-spacing:0.05em'>"
                    "🔀 HYBRID 17+33 kp (YOLO + MediaPipe)</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<span style='background:#718096;color:white;border-radius:6px;"
                    "padding:2px 10px;font-size:0.72rem;font-weight:700;letter-spacing:0.05em'>"
                    "YOLO-only 17 kp</span>",
                    unsafe_allow_html=True
                )

            # Joint angles
            st.markdown("**📐 Joint Angles**")
            neck_val = angles.get("neck") or angles.get("neck_flexion")
            wrist_val = angles.get("wrist_deviation")
            angle_map = {
                "Trunk":    angles.get("trunk") or angles.get("trunk_flexion"),
                "Neck":     neck_val,
                "Shoulder": angles.get("shoulder_elevation"),
                "Elbow":    angles.get("elbow_flexion"),
                "Knee":     angles.get("knee_flexion"),
                "Wrist":    wrist_val,
            }
            rows_a = list(angle_map.items())
            for i in range(0, len(rows_a), 2):
                ca, cb = st.columns(2)
                for col_j, (jname, jval) in zip([ca, cb], rows_a[i:i+2]):
                    with col_j:
                        if jval:
                            vstr = f"{jval}°"
                        elif jname in ("Neck", "Wrist"):
                            vstr = "N/A (YOLO-only)"
                        else:
                            vstr = "N/A"
                        st.markdown(f"<small style='color:#718096'>{jname}</small><br><b>{vstr}</b>",
                                    unsafe_allow_html=True)

            st.markdown("---")

            # Joint risk bars  (REBA weights: Trunk 35% | Neck 15% | Shoulder 25% | Elbow 10% | Knee 10% | Wrist 5%)
            st.markdown("**📊 Joint Risk Scores**")
            score_map = {
                "Trunk (35%)":    joints.get("trunk_flexion",     joints.get("trunk",    0)),
                "Neck (15%)":     joints.get("neck_flexion",      joints.get("neck",     0)),
                "Shoulder (25%)": joints.get("shoulder_elevation", 0),
                "Elbow (10%)":    joints.get("elbow_flexion",     0),
                "Knee (10%)":     joints.get("knee_flexion",      0),
                "Wrist (5%)":     joints.get("wrist_deviation",   joints.get("wrist_left", 0)),
            }
            for lbl, js in score_map.items():
                pct = float(js)
                st.markdown(f"<small>{lbl} — <b>{round(pct*10,1)}/10</b></small>",
                            unsafe_allow_html=True)
                st.progress(pct)

            st.markdown("---")

            # Frozen zones
            if frozen:
                st.markdown("**⚠️ Frozen Zones (low confidence)**")
                ztags = " ".join([f'<span class="ztag">{z}</span>' for z in frozen])
                st.markdown(ztags, unsafe_allow_html=True)
                st.markdown(f"<small style='color:#A0AEC0'>{sel.get('filter_reason','')}</small>",
                            unsafe_allow_html=True)
            else:
                st.markdown("**✅ All body zones visible**")

            # ── Worst-frame Audit Trail ───────────────────────────────────────
            all_scores  = sel.get("all_frame_scores", [])
            frame_count = sel.get("frame_count", 1)
            if all_scores and frame_count > 1:
                st.markdown(f"**🎞️ Frame Scoring ({frame_count} frames — worst selected)**")
                rows_fs = ""
                for fp, sc, col in all_scores:
                    fname = Path(fp).name
                    is_best = (fp == sel.get("img_path", ""))
                    cmark = "⭐" if is_best else ""
                    sc_str = f"{sc:.2f}" if sc is not None else "—"
                    col_dot = {"GREEN":"🟢","AMBER":"🟡","RED":"🔴"}.get(col, "⚪")
                    rows_fs += (
                        f"<tr style='{'font-weight:700;background:#F0FFF4' if is_best else ''}'>"
                        f"<td>{cmark} {fname}</td>"
                        f"<td style='text-align:center'>{col_dot} {col}</td>"
                        f"<td style='text-align:center'>{sc_str}</td>"
                        f"</tr>"
                    )
                st.markdown(f"""
                <table style='width:100%;font-size:0.76rem;border-collapse:collapse;margin-bottom:10px;'>
                    <tr style='color:#888;border-bottom:1px solid #ddd;'>
                        <th style='text-align:left'>Frame</th>
                        <th style='text-align:center'>Risk</th>
                        <th style='text-align:center'>Score</th>
                    </tr>
                    {rows_fs}
                </table>
                <small style='color:#A0AEC0'>⭐ = worst-risk frame shown in card</small>
                """, unsafe_allow_html=True)

            # ── Occlusion Handling Proof Panel ────────────────────────────────
            est        = sel.get("estimated_kp", [])
            occ_report = sel.get("occ_report", {})
            occ_path   = sel.get("occ_report_path")
            res        = occ_report.get("results", {})
            orig_vis   = res.get("original_visible_count", "—")
            final_vis  = res.get("final_visible_count",    "—")
            frame_gain = res.get("frame_gain",             0)
            est_detail = res.get("estimated_keypoints",    [])

            # Always show the occlusion section header
            st.markdown("**🔬 Occlusion Handling**")

            if est:
                # Per-keypoint rows
                rows_html = ""
                for e in est_detail:
                    rows_html += (
                        f"<tr>"
                        f"<td>{e['name'].replace('_', ' ').title()}</td>"
                        f"<td style='color:#888'>← {e['mirrored_from_name'].replace('_',' ').title()}</td>"
                        f"<td>{e['source_conf']:.2f}</td>"
                        f"<td><b>{e['estimated_conf']:.2f}</b></td>"
                        f"</tr>"
                    )
                st.markdown(f"""
                <div style="background:#FFF8E1;border-left:4px solid #FFB300;
                            padding:10px 14px;border-radius:8px;margin-top:6px;
                            font-size:0.78rem;color:#5D4037;">
                    <b>🔧 {len(est)} keypoint(s) estimated via body-symmetry mirroring</b><br>
                    <span style="color:#7B5E00">
                        Visible keypoints: {orig_vis} → {final_vis}
                        &nbsp;|&nbsp; Frame gain: <b>+{frame_gain}</b>
                    </span>
                    <table style="width:100%;margin-top:8px;border-collapse:collapse;
                                  font-size:0.76rem;">
                        <tr style="color:#888;border-bottom:1px solid #e0cfa0;">
                            <th style="text-align:left">Estimated</th>
                            <th style="text-align:left">Source</th>
                            <th>Src conf</th>
                            <th>Est conf</th>
                        </tr>
                        {rows_html}
                    </table>
                    <div style="margin-top:6px;color:#A0856A;font-size:0.72rem;">
                        Confidence reduced × 0.60 · x-coordinate reflected across image midline
                    </div>
                </div>""", unsafe_allow_html=True)
                if occ_path:
                    st.caption(f"📄 Proof file: `{occ_path}`")
            else:
                st.markdown(
                    '<div style="background:#F1F8E9;border-left:4px solid #66BB6A;'
                    'padding:8px 12px;border-radius:6px;font-size:0.78rem;color:#2E7D32;">'
                    '✅ No occlusion estimation needed — all bilateral keypoints detected'
                    '</div>', unsafe_allow_html=True)

            st.markdown("---")

            adv_level, adv_text = get_advisory(sel)
            st.markdown(f'<div class="{ADV_CLASS[adv_level]}">{adv_text}</div>',
                        unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About the F3 Pipeline")
    st.markdown("""
    ### Warehouse Intelligence System — Function 3
    **Project:** J26-DS-352 | SLIIT Final Year Research

    ---

    ### Pipeline Architecture

    ```
    Worker Crop / Frame Image
            ↓
    YOLOv11m-pose  →  17 keypoints [x, y, confidence]
            ↓
    Novelty 1: Keypoint Confidence Filter
    • Threshold: conf > 0.30
    • Body zones: upper / core / lower
    • Freeze unreliable zones
            ↓
    Novelty 2: Ergonomic Composite Risk Scorer
    • Joint angles: trunk, shoulder, elbow, knee
    • REBA-based scoring (1–10 per joint)
    • Weighted composite:
        Trunk 40% + Shoulder 30% + Elbow 15% + Knee 15%
    • GREEN ≤ 3.0 | AMBER ≤ 5.5 | RED > 5.5
            ↓
    Novelty 3: Demographic MLP  [pending training data]
    • Age group + Gender classification
    • HOG features (3780-dim) → pure NumPy MLP
    ```

    ---

    ### Risk Colour Guide

    | Colour | Score | Action |
    |--------|-------|--------|
    | 🟢 GREEN | ≤ 3.0 | Safe — no action needed |
    | 🟡 AMBER | 3.1–5.5 | Monitor — consider task rotation |
    | 🔴 RED | > 5.5 | Intervene — rest or reassign |

    ---

    ### Data Sources (priority order)
    1. `data/worker-crops/` — manually curated tight crops
    2. `data/crops/` — auto-cropped from YOLO detections
    3. `data/frames/` — raw extracted video frames

    ---

    ### References
    - Hignett & McAtamney (2000) — REBA, *Applied Ergonomics* 31(2):201-205
    - Wang et al. (2023) — Keypoint confidence thresholds
    - Xu et al. (2022) — ViTPose keypoint filtering
    """)