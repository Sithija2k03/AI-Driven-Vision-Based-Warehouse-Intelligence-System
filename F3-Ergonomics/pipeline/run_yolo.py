import cv2
import os
from ultralytics import YOLO

FRAMES_FOLDER = r"D:\F3-Project\data\frames"
OUTPUT_FOLDER = r"D:\F3-Project\data\annotated"

model = YOLO("yolov8n-pose.pt")  # downloads automatically first time

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

worker_folders = sorted(os.listdir(FRAMES_FOLDER))

for worker in worker_folders:
    worker_frame_dir = os.path.join(FRAMES_FOLDER, worker)
    if not os.path.isdir(worker_frame_dir):
        continue

    out_dir = os.path.join(OUTPUT_FOLDER, worker)
    os.makedirs(out_dir, exist_ok=True)

    frames = sorted(f for f in os.listdir(worker_frame_dir) if f.endswith(".jpg"))
    print(f"\n[{worker}] Processing {len(frames)} frames...")

    detected = 0
    for frame_file in frames:
        frame_path = os.path.join(worker_frame_dir, frame_file)
        frame = cv2.imread(frame_path)

        results = model(frame, conf=0.3)

        annotated = results[0].plot()
        out_path = os.path.join(out_dir, frame_file)
        cv2.imwrite(out_path, annotated)

        num_people = len(results[0].boxes)
        if num_people > 0:
            detected += 1

    print(f"[{worker}] Done — worker detected in {detected}/{len(frames)} frames")

print("\nAll workers processed. Check D:\\F3-Project\\data\\annotated\\")