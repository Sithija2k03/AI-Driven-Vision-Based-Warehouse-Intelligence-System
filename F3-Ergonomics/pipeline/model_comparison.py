import cv2
import os
import time
from ultralytics import YOLO

FRAMES_FOLDER = r"D:\F3-Project\data\frames"
OUTPUT_FOLDER = r"D:\F3-Project\data\model_comparison"

MODELS = [
    ("YOLOv8n-pose",  "yolov8n-pose.pt"),
    ("YOLOv8s-pose",  "yolov8s-pose.pt"),
    ("YOLOv8m-pose",  "yolov8m-pose.pt"),
    ("YOLOv11s-pose", "yolo11s-pose.pt"),
]
CONF_THRESHOLD = 0.3

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Collect all frame paths
all_frames = []
for worker in sorted(os.listdir(FRAMES_FOLDER)):
    worker_dir = os.path.join(FRAMES_FOLDER, worker)
    if not os.path.isdir(worker_dir):
        continue
    for f in sorted(os.listdir(worker_dir)):
        if f.endswith(".jpg"):
            all_frames.append(os.path.join(worker_dir, f))

print(f"Total frames to test: {len(all_frames)}")
print("=" * 60)

results_summary = []

for model_name, weights in MODELS:
    print(f"\nLoading {model_name}...")
    model = YOLO(weights)

    out_dir = os.path.join(OUTPUT_FOLDER, model_name)
    os.makedirs(out_dir, exist_ok=True)

    total_frames     = 0
    frames_detected  = 0
    total_confidence = 0.0
    total_people     = 0
    total_keypoints  = 0
    total_time_ms    = 0.0

    for frame_path in all_frames:
        frame = cv2.imread(frame_path)
        total_frames += 1

        start = time.perf_counter()
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_time_ms += elapsed_ms

        boxes = results[0].boxes
        keypoints = results[0].keypoints

        if len(boxes) > 0:
            frames_detected += 1
            for box in boxes:
                total_confidence += float(box.conf[0])
                total_people += 1

        if keypoints is not None:
            for kp in keypoints.xy:
                visible = sum(1 for k in kp if k[0] > 0 and k[1] > 0)
                total_keypoints += visible

        # Save annotated frame
        annotated = results[0].plot()
        out_name = os.path.join(out_dir, os.path.basename(frame_path))
        cv2.imwrite(out_name, annotated)

    detection_rate   = (frames_detected / total_frames * 100) if total_frames > 0 else 0
    avg_confidence   = (total_confidence / total_people) if total_people > 0 else 0
    avg_keypoints    = (total_keypoints / total_people) if total_people > 0 else 0
    avg_speed        = (total_time_ms / total_frames) if total_frames > 0 else 0

    results_summary.append({
        "model":          model_name,
        "detection_rate": detection_rate,
        "avg_confidence": avg_confidence,
        "avg_keypoints":  avg_keypoints,
        "avg_speed_ms":   avg_speed,
        "total_people":   total_people,
    })

    print(f"[{model_name}] Detection rate : {detection_rate:.1f}%")
    print(f"[{model_name}] Avg confidence : {avg_confidence:.3f}")
    print(f"[{model_name}] Avg keypoints  : {avg_keypoints:.1f} / 17")
    print(f"[{model_name}] Avg speed      : {avg_speed:.1f} ms/frame")
    print(f"[{model_name}] Total people   : {total_people}")

# Final comparison table
print("\n")
print("=" * 60)
print("FINAL COMPARISON TABLE")
print("=" * 60)
print(f"{'Model':<18} {'Detection%':>10} {'Avg Conf':>10} {'Avg KP':>8} {'Speed ms':>10}")
print("-" * 60)
for r in results_summary:
    print(f"{r['model']:<18} {r['detection_rate']:>9.1f}% {r['avg_confidence']:>10.3f} {r['avg_keypoints']:>8.1f} {r['avg_speed_ms']:>9.1f}ms")
print("=" * 60)

# Save results to text file
results_path = os.path.join(OUTPUT_FOLDER, "comparison_results.txt")
with open(results_path, "w") as f:
    f.write("F3 YOLO MODEL COMPARISON RESULTS\n")
    f.write("Warehouse Intelligence System | J26-DS-352\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'Model':<18} {'Detection%':>10} {'Avg Conf':>10} {'Avg KP':>8} {'Speed ms':>10}\n")
    f.write("-" * 60 + "\n")
    for r in results_summary:
        f.write(f"{r['model']:<18} {r['detection_rate']:>9.1f}% {r['avg_confidence']:>10.3f} {r['avg_keypoints']:>8.1f} {r['avg_speed_ms']:>9.1f}ms\n")
    f.write("=" * 60 + "\n")

print(f"\nResults saved to: {results_path}")
print("Annotated frames saved to: D:\\F3-Project\\data\\model_comparison\\")