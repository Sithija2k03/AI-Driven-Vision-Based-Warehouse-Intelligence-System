

import os
import cv2
import csv
import time
import argparse
import numpy as np
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
PIPELINE_DIR  = r"D:\F3-Project\data\media-pipeline"
RESULTS_CSV   = r"D:\F3-Project\outputs\benchmark_results.csv"
SAMPLES_DIR   = r"D:\F3-Project\outputs\benchmark_samples\MediaPipe"

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--frames_per_worker", type=int, default=10,
                    help="Frames to sample per worker folder (default: 10, same as YOLO benchmark)")
parser.add_argument("--conf", type=float, default=0.3,
                    help="Min detection confidence for MediaPipe (default: 0.3)")
args = parser.parse_args()

FRAMES_PER_WORKER = args.frames_per_worker
CONF_THRESHOLD    = args.conf

# ── MediaPipe skeleton pairs (33 landmarks) ────────────────────────────────────
MP_PAIRS = [
    (0,1),(1,2),(2,3),(3,7),          # face left
    (0,4),(4,5),(5,6),(6,8),          # face right
    (11,12),                          # shoulders
    (11,13),(13,15),(15,17),(15,19),  # left arm
    (12,14),(14,16),(16,18),(16,20),  # right arm
    (11,23),(12,24),(23,24),          # torso
    (23,25),(25,27),(27,29),(27,31),  # left leg
    (24,26),(26,28),(28,30),(28,32),  # right leg
]

def draw_mp_pose(frame, landmarks, h, w, color=(0,255,0)):
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h), lm.visibility))
    for i, (x, y, v) in enumerate(pts):
        if v > 0.3:
            cv2.circle(frame, (x, y), 4, color, -1)
    for a, b in MP_PAIRS:
        if a < len(pts) and b < len(pts):
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va > 0.3 and vb > 0.3:
                cv2.line(frame, (xa, ya), (xb, yb), color, 2)
    return frame

# ── Collect same sample frames as YOLO benchmark ──────────────────────────────
def collect_sample_frames(n_per_worker):
    samples = []
    if not os.path.isdir(PIPELINE_DIR):
        print(f"[ERROR] media-pipeline folder not found: {PIPELINE_DIR}")
        return samples
    for worker in sorted(os.listdir(PIPELINE_DIR)):
        worker_path = os.path.join(PIPELINE_DIR, worker)
        if not os.path.isdir(worker_path):
            continue
        frames = sorted([f for f in os.listdir(worker_path)
                         if f.lower().endswith(".jpg")])
        if not frames:
            continue
        indices = np.linspace(0, len(frames)-1,
                              min(n_per_worker, len(frames)), dtype=int)
        for i in indices:
            samples.append((worker, os.path.join(worker_path, frames[i])))
    return samples

# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    print(f"\nF3 MediaPipe Benchmark")
    print(f"======================")
    print(f"Source  : {PIPELINE_DIR}")
    print(f"Output  : {RESULTS_CSV}  (append mode)")
    print(f"Frames per worker: {FRAMES_PER_WORKER}\n")

    # Import mediapipe here so a missing install gives a clear message
    try:
        import mediapipe as mp
    except ImportError:
        print("[ERROR] mediapipe not installed.")
        print("  Run:  pip install mediapipe")
        return

    mp_pose     = mp.solutions.pose
    mp_drawing  = mp.solutions.drawing_utils

    # Collect frames
    samples = collect_sample_frames(FRAMES_PER_WORKER)
    if not samples:
        print("[ERROR] No frames found. Run media_pipeline.py first.")
        return
    print(f"Total frames to test: {len(samples)}\n")

    os.makedirs(SAMPLES_DIR, exist_ok=True)

    total_frames       = 0
    frames_with_person = 0
    total_persons      = 0
    conf_scores        = []      # detection confidence = avg landmark visibility
    kp_conf_scores     = []      # per-person avg visibility
    inference_times_ms = []
    saved_samples      = 0

    # MediaPipe Pose runs one person per call (single-person model)
    # We use it in static_image_mode for frame-by-frame processing
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,          # 0=lite, 1=full, 2=heavy
        min_detection_confidence=CONF_THRESHOLD,
        min_tracking_confidence=CONF_THRESHOLD,
    ) as pose:

        for worker_name, frame_path in samples:
            img = cv2.imread(frame_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            total_frames += 1

            # MediaPipe expects RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            result = pose.process(img_rgb)
            t1 = time.perf_counter()

            inference_ms = (t1 - t0) * 1000
            inference_times_ms.append(inference_ms)

            if result.pose_landmarks:
                frames_with_person += 1
                total_persons      += 1   # MediaPipe detects 1 person per frame

                # Visibility scores as confidence proxy
                vis_scores = [lm.visibility for lm in result.pose_landmarks.landmark]
                avg_vis    = float(np.mean(vis_scores))
                conf_scores.append(avg_vis)
                kp_conf_scores.append(avg_vis)

                # Save annotated sample (first 5)
                if saved_samples < 5:
                    ann = img.copy()
                    draw_mp_pose(ann, result.pose_landmarks.landmark, h, w)
                    label = f"MediaPipe | 1 person | {inference_ms:.1f}ms"
                    cv2.putText(ann, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    out_name = f"{worker_name}_sample{saved_samples+1}.jpg"
                    cv2.imwrite(os.path.join(SAMPLES_DIR, out_name), ann,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    saved_samples += 1

            if total_frames % 20 == 0:
                print(f"  Processed {total_frames}/{len(samples)} frames...")

    # ── Metrics ────────────────────────────────────────────────────────────────
    detection_rate = (frames_with_person / total_frames * 100) if total_frames > 0 else 0
    avg_det_conf   = float(np.mean(conf_scores))    if conf_scores    else 0.0
    avg_kp_conf    = float(np.mean(kp_conf_scores)) if kp_conf_scores else 0.0
    avg_infer_ms   = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    fps            = 1000 / avg_infer_ms if avg_infer_ms > 0 else 0.0

    print(f"\n  Results for MediaPipe Pose:")
    print(f"    Frames processed      : {total_frames}")
    print(f"    Detection rate        : {detection_rate:.1f}%")
    print(f"    Total persons found   : {total_persons}")
    print(f"    Avg detection conf    : {avg_det_conf:.3f}  (landmark visibility)")
    print(f"    Avg keypoint conf     : {avg_kp_conf:.3f}")
    print(f"    Avg inference time    : {avg_infer_ms:.1f} ms/frame")
    print(f"    Throughput            : {fps:.1f} FPS")

    new_row = {
        "model"              : "MediaPipe-Pose",
        "model_file"         : "mediapipe (pip)",
        "pose_support"       : "Yes",
        "frames_processed"   : total_frames,
        "detection_rate_pct" : round(detection_rate, 2),
        "total_persons"      : total_persons,
        "avg_det_confidence" : round(avg_det_conf, 4),
        "avg_kp_confidence"  : round(avg_kp_conf, 4),
        "avg_inference_ms"   : round(avg_infer_ms, 2),
        "throughput_fps"     : round(fps, 2),
        "benchmarked_at"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Append to existing CSV ─────────────────────────────────────────────────
    if not os.path.isfile(RESULTS_CSV):
        print(f"\n[WARN] {RESULTS_CSV} not found — creating new file.")
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(new_row.keys()))
            writer.writeheader()
            writer.writerow(new_row)
    else:
        # Read existing to get fieldnames, then append
        with open(RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        fieldnames = list(existing[0].keys()) if existing else list(new_row.keys())

        # Remove any existing MediaPipe row (avoid duplicates on re-run)
        existing = [r for r in existing if r.get("model") != "MediaPipe-Pose"]
        existing.append(new_row)

        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MEDIAPIPE BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Det.Rate':>9} {'Det.Conf':>9} {'KP.Conf':>8} {'ms/frame':>10} {'FPS':>7}")
    print(f"{'-'*65}")
    print(f"{'MediaPipe-Pose':<20} {detection_rate:>8.1f}% "
          f"{avg_det_conf:>9.3f} {avg_kp_conf:>8.3f} "
          f"{avg_infer_ms:>9.1f}ms {fps:>6.1f}")
    print(f"\nRow appended to  → {RESULTS_CSV}")
    print(f"Annotated samples→ {SAMPLES_DIR}")
    print(f"\nNOTE: MediaPipe detects 1 person per frame (single-person model).")
    print(f"      YOLO detects multiple persons per frame.")
    print(f"      This is a key limitation to state in your comparison table.")


if __name__ == "__main__":
    run()