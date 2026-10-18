

import os
import cv2
import csv
import time
import argparse
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
PIPELINE_DIR  = r"D:\F3-Project\data\media-pipeline"
OUTPUT_DIR    = r"D:\F3-Project\outputs"
RESULTS_CSV   = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
SAMPLES_DIR   = os.path.join(OUTPUT_DIR, "benchmark_samples")

# ── Models to compare ─────────────────────────────────────────────────────────
# Ultralytics auto-downloads weights on first run
MODELS = {
    "YOLOv8m-pose" : "yolov8m-pose.pt",
    "YOLOv9s"      : "yolov9s.pt",        # detection only (no pose variant widely available)
    "YOLOv10s"     : "yolov10s.pt",       # detection only
    "YOLOv11m-pose": "yolo11m-pose.pt",   # pose
}

# Keypoint model names (for checking if pose output is available)
POSE_MODELS = {"YOLOv8m-pose", "YOLOv11m-pose"}

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--frames_per_worker", type=int, default=10,
                    help="How many frames to sample per worker folder (default: 10)")
parser.add_argument("--conf", type=float, default=0.3,
                    help="Detection confidence threshold (default: 0.3)")
args = parser.parse_args()

FRAMES_PER_WORKER = args.frames_per_worker
CONF_THRESHOLD    = args.conf

# ── Skeleton drawing ───────────────────────────────────────────────────────────
COCO_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),          # head
    (5,6),(5,7),(7,9),(6,8),(8,10),   # arms
    (5,11),(6,12),(11,12),            # torso
    (11,13),(13,15),(12,14),(14,16),  # legs
]

def draw_pose(frame, keypoints, color=(0,255,0)):
    """Draw skeleton on frame from keypoints array (shape: 17x3)."""
    for i, (x, y, c) in enumerate(keypoints):
        if c > 0.3:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for a, b in COCO_PAIRS:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca > 0.3 and cb > 0.3:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)
    return frame

# ── Sample frames from pipeline output ────────────────────────────────────────
def collect_sample_frames(n_per_worker):
    """Return list of (worker_name, frame_path) tuples."""
    samples = []
    if not os.path.isdir(PIPELINE_DIR):
        print(f"[ERROR] media-pipeline folder not found: {PIPELINE_DIR}")
        return samples

    for worker in sorted(os.listdir(PIPELINE_DIR)):
        worker_path = os.path.join(PIPELINE_DIR, worker)
        if not os.path.isdir(worker_path):
            continue
        frames = sorted([
            f for f in os.listdir(worker_path)
            if f.lower().endswith(".jpg")
        ])
        # Evenly sample n_per_worker frames
        if len(frames) == 0:
            continue
        indices = np.linspace(0, len(frames)-1, min(n_per_worker, len(frames)), dtype=int)
        for i in indices:
            samples.append((worker, os.path.join(worker_path, frames[i])))

    return samples

# ── Run one model over all samples ────────────────────────────────────────────
def benchmark_model(model_name, model_path, samples, save_samples=True):
    print(f"\n{'='*55}")
    print(f"  Benchmarking: {model_name}")
    print(f"{'='*55}")

    # Load model — runs on GPU if CUDA available, else CPU
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"  [ERROR] Could not load {model_path}: {e}")
        return None

    is_pose = model_name in POSE_MODELS

    sample_out_dir = os.path.join(SAMPLES_DIR, model_name.replace(" ", "_"))
    if save_samples:
        os.makedirs(sample_out_dir, exist_ok=True)

    total_frames        = 0
    frames_with_person  = 0
    total_persons       = 0
    conf_scores         = []
    kp_conf_scores      = []
    inference_times_ms  = []
    saved_samples       = 0

    for worker_name, frame_path in samples:
        img = cv2.imread(frame_path)
        if img is None:
            continue

        total_frames += 1

        # ── Inference ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = model(img, conf=CONF_THRESHOLD, verbose=False)
        t1 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000
        inference_times_ms.append(inference_ms)

        result = results[0]
        boxes  = result.boxes

        n_persons = len(boxes) if boxes is not None else 0
        total_persons += n_persons

        if n_persons > 0:
            frames_with_person += 1
            # Detection confidence
            for conf in boxes.conf.cpu().numpy():
                conf_scores.append(float(conf))

            # Keypoint confidence (pose models only)
            if is_pose and result.keypoints is not None:
                kps = result.keypoints.conf  # shape: (n_persons, 17)
                if kps is not None:
                    for person_kp_conf in kps.cpu().numpy():
                        kp_conf_scores.append(float(np.mean(person_kp_conf)))

        # ── Save annotated sample (first 5 per model) ─────────────────────────
        if save_samples and saved_samples < 5:
            ann = img.copy()
            # Draw boxes
            if boxes is not None:
                for box in boxes.xyxy.cpu().numpy():
                    x1,y1,x2,y2 = map(int, box)
                    cv2.rectangle(ann, (x1,y1), (x2,y2), (0,200,255), 2)
            # Draw skeleton (pose models)
            if is_pose and result.keypoints is not None and result.keypoints.xy is not None:
                kp_xy   = result.keypoints.xy.cpu().numpy()    # (n,17,2)
                kp_conf = result.keypoints.conf.cpu().numpy()  # (n,17)
                for i in range(len(kp_xy)):
                    kp = np.concatenate([kp_xy[i], kp_conf[i:i+1].T], axis=1)  # (17,3)
                    ann = draw_pose(ann, kp, color=(0,255,0))

            # Label
            label = f"{model_name} | {n_persons} person(s) | {inference_ms:.1f}ms"
            cv2.putText(ann, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            out_name = f"{worker_name}_sample{saved_samples+1}.jpg"
            cv2.imwrite(os.path.join(sample_out_dir, out_name), ann,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_samples += 1

        # Progress
        if total_frames % 20 == 0:
            print(f"  Processed {total_frames}/{len(samples)} frames...")

    # ── Compute summary metrics ────────────────────────────────────────────────
    detection_rate    = (frames_with_person / total_frames * 100) if total_frames > 0 else 0
    avg_det_conf      = np.mean(conf_scores)    if conf_scores    else 0.0
    avg_kp_conf       = np.mean(kp_conf_scores) if kp_conf_scores else None
    avg_infer_ms      = np.mean(inference_times_ms) if inference_times_ms else 0.0
    fps_throughput    = 1000 / avg_infer_ms if avg_infer_ms > 0 else 0

    print(f"\n  Results for {model_name}:")
    print(f"    Frames processed      : {total_frames}")
    print(f"    Detection rate        : {detection_rate:.1f}%")
    print(f"    Total persons found   : {total_persons}")
    print(f"    Avg detection conf    : {avg_det_conf:.3f}")
    if avg_kp_conf is not None:
        print(f"    Avg keypoint conf     : {avg_kp_conf:.3f}")
    print(f"    Avg inference time    : {avg_infer_ms:.1f} ms/frame")
    print(f"    Throughput            : {fps_throughput:.1f} FPS")

    return {
        "model"              : model_name,
        "model_file"         : model_path,
        "pose_support"       : "Yes" if is_pose else "No",
        "frames_processed"   : total_frames,
        "detection_rate_pct" : round(detection_rate, 2),
        "total_persons"      : total_persons,
        "avg_det_confidence" : round(avg_det_conf, 4),
        "avg_kp_confidence"  : round(avg_kp_conf, 4) if avg_kp_conf is not None else "N/A",
        "avg_inference_ms"   : round(avg_infer_ms, 2),
        "throughput_fps"     : round(fps_throughput, 2),
        "benchmarked_at"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    print(f"\nF3 Model Benchmark")
    print(f"==================")
    print(f"Source : {PIPELINE_DIR}")
    print(f"Models : {list(MODELS.keys())}")
    print(f"Frames per worker: {FRAMES_PER_WORKER}")

    # Collect frames
    samples = collect_sample_frames(FRAMES_PER_WORKER)
    if not samples:
        print("[ERROR] No frames found. Run media_pipeline.py first.")
        return

    print(f"Total frames to test: {len(samples)} ({FRAMES_PER_WORKER} per worker)")

    # Run each model
    all_results = []
    for model_name, model_path in MODELS.items():
        result = benchmark_model(model_name, model_path, samples)
        if result:
            all_results.append(result)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved → {RESULTS_CSV}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"BENCHMARK COMPLETE — SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<18} {'Det.Rate':>9} {'Det.Conf':>9} {'KP.Conf':>9} {'ms/frame':>10} {'FPS':>7}")
    print(f"{'-'*64}")
    for r in all_results:
        kp = f"{r['avg_kp_confidence']}" if r['avg_kp_confidence'] != "N/A" else "  N/A"
        print(f"{r['model']:<18} {r['detection_rate_pct']:>8.1f}% "
              f"{r['avg_det_confidence']:>9.3f} {kp:>9} "
              f"{r['avg_inference_ms']:>9.1f}ms {r['throughput_fps']:>6.1f}")
    print(f"\nAnnotated samples → {SAMPLES_DIR}")
    print(f"Results CSV       → {RESULTS_CSV}")


if __name__ == "__main__":
    run_benchmark()