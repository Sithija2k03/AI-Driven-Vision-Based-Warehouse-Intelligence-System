

import json
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ── YOLO-17 keypoint name table ───────────────────────────────────────────────
KP_NAMES = {
    0:  "nose",
    1:  "left_eye",
    2:  "right_eye",
    3:  "left_ear",
    4:  "right_ear",
    5:  "left_shoulder",
    6:  "right_shoulder",
    7:  "left_elbow",
    8:  "right_elbow",
    9:  "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Bilateral mirror pairs: {missing_idx: source_idx}
# Only body keypoints are mirrored; midline keypoints (nose, eyes, ears) are
# excluded — their contralateral geometry is unreliable across person orientations.
MIRROR_PAIRS: Dict[int, int] = {
    5: 6,   6: 5,    # shoulders
    7: 8,   8: 7,    # elbows
    9: 10, 10: 9,    # wrists
    11: 12, 12: 11,  # hips
    13: 14, 14: 13,  # knees
    15: 16, 16: 15,  # ankles
}

# Minimum confidence of the SOURCE keypoint to trust a mirror estimation
MIRROR_SOURCE_CONF_THRESHOLD: float = 0.30

# Estimated confidence = source_conf × CONF_REDUCTION_FACTOR
CONF_REDUCTION_FACTOR: float = 0.60


# ── Core function ─────────────────────────────────────────────────────────────
def apply_occlusion_handling(
    keypoints:   List[List[float]],
    img_width:   int,
    worker_id:   str = "unknown",
    frame_label: str = "frame",
) -> Tuple[List[List[float]], List[int], Dict[str, Any]]:
    """
    Estimate missing keypoints via bilateral body-symmetry mirroring.

    Parameters
    ----------
    keypoints   : list of 17 [x, y, conf] entries (from confidence_filter output)
    img_width   : pixel width of the source image (used to mirror x coordinates)
    worker_id   : string identifier for the worker (used in the report)
    frame_label : descriptive label for the frame (e.g. 'rep_frame', 'frame_003')

    Returns
    -------
    corrected_kps     : new list of 17 [x, y, conf] — same structure, but
                        missing keypoints filled where possible
    estimated_indices : list of int — indices of keypoints that were estimated
    report_dict       : dict — full audit trail of what was done and why
    """
    # Deep-copy so we never mutate the caller's list
    corrected = [list(kp) for kp in keypoints]

    original_visible = sum(1 for kp in corrected if kp[2] > 0.01)

    estimated_indices: List[int] = []
    estimated_detail:  List[Dict[str, Any]] = []

    for idx in range(len(corrected)):
        kp = corrected[idx]

        # Skip if already visible
        if kp[2] >= 0.01:
            continue

        # Check if this index has a mirror pair
        mirror_idx = MIRROR_PAIRS.get(idx)
        if mirror_idx is None:
            continue

        source_kp = corrected[mirror_idx]

        # Only mirror if the source keypoint is reliably detected
        if source_kp[2] < MIRROR_SOURCE_CONF_THRESHOLD:
            continue

        # Reflect x across the image midline; y is shared between mirrored pairs
        est_x    = img_width - source_kp[0]
        est_y    = source_kp[1]
        est_conf = source_kp[2] * CONF_REDUCTION_FACTOR

        corrected[idx][0] = est_x
        corrected[idx][1] = est_y
        corrected[idx][2] = est_conf

        estimated_indices.append(idx)
        estimated_detail.append({
            "index":              idx,
            "name":               KP_NAMES.get(idx, f"kp_{idx}"),
            "mirrored_from_index": mirror_idx,
            "mirrored_from_name": KP_NAMES.get(mirror_idx, f"kp_{mirror_idx}"),
            "source_conf":        round(float(source_kp[2]), 4),
            "estimated_x":        round(float(est_x), 2),
            "estimated_y":        round(float(est_y), 2),
            "estimated_conf":     round(float(est_conf), 4),
            "conf_reduction":     f"× {CONF_REDUCTION_FACTOR} (flagged as estimated)",
        })

    final_visible = sum(1 for kp in corrected if kp[2] > 0.01)

    report_dict: Dict[str, Any] = {
        "meta": {
            "module":       "occlusion_handler.py",
            "version":      "1.0.0",
            "project":      "F3 — Warehouse Intelligence System J26-DS-352",
            "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
            "worker_id":    worker_id,
            "frame_label":  frame_label,
            "img_width_px": img_width,
        },
        "config": {
            "mirror_source_conf_threshold": MIRROR_SOURCE_CONF_THRESHOLD,
            "conf_reduction_factor":        CONF_REDUCTION_FACTOR,
            "mirrored_pairs": [
                f"{KP_NAMES[k]} ↔ {KP_NAMES[v]}"
                for k, v in MIRROR_PAIRS.items()
                if k < v   # deduplicate symmetric pairs
            ],
        },
        "results": {
            "original_visible_count": original_visible,
            "final_visible_count":    final_visible,
            "estimated_count":        len(estimated_indices),
            "frame_gain":             final_visible - original_visible,
            "estimated_keypoints":    estimated_detail,
        },
        "summary": (
            f"{len(estimated_indices)} keypoint(s) estimated via body-symmetry mirroring "
            f"({original_visible} → {final_visible} visible, "
            f"+{final_visible - original_visible} frame_gain)"
        ),
    }

    return corrected, estimated_indices, report_dict


# ── Report persistence ────────────────────────────────────────────────────────
def save_report(
    report:   Dict[str, Any],
    out_dir:  str = r"D:\F3-Project\pipeline\occlusion_reports",
    filename: Optional[str] = None,
) -> str:
    """
    Write the report dict to a JSON file.

    File name (if not given):
        occlusion_<worker_id>_<frame_label>_<YYYYMMDD_HHMMSS>.json

    Returns the absolute path of the written file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        worker_id = report["meta"].get("worker_id", "unknown").replace(" ", "-")
        frame_lbl = report["meta"].get("frame_label", "frame").replace(" ", "-")
        filename  = f"occlusion_{worker_id}_{frame_lbl}_{ts}.json"

    file_path = out_path / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(file_path)


# ── Batch summary (aggregate across workers) ──────────────────────────────────
def build_batch_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of per-worker report dicts, produce an aggregate summary
    suitable for a paper table or CSV row.

    Returns a dict with total and per-worker statistics.
    """
    workers = []
    total_original = 0
    total_final    = 0
    total_estimated = 0

    for r in reports:
        res = r["results"]
        meta = r["meta"]
        workers.append({
            "worker_id":              meta.get("worker_id"),
            "frame_label":            meta.get("frame_label"),
            "original_visible_count": res["original_visible_count"],
            "final_visible_count":    res["final_visible_count"],
            "estimated_count":        res["estimated_count"],
            "frame_gain":             res["frame_gain"],
            "estimated_keypoints":    [e["name"] for e in res["estimated_keypoints"]],
        })
        total_original  += res["original_visible_count"]
        total_final     += res["final_visible_count"]
        total_estimated += res["estimated_count"]

    n = len(reports)
    return {
        "total_workers_processed": n,
        "total_original_visible":  total_original,
        "total_final_visible":     total_final,
        "total_estimated":         total_estimated,
        "avg_frame_gain":          round((total_final - total_original) / n, 2) if n else 0,
        "workers":                 workers,
    }


def save_batch_summary(
    batch_summary: Dict[str, Any],
    out_dir:       str = r"D:\F3-Project\pipeline\occlusion_reports",
    filename:      str = "occlusion_batch_summary.json",
) -> str:
    """Write the batch summary to a JSON file. Returns absolute path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    return str(file_path)