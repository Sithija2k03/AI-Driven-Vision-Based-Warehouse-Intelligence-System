import cv2
import os
from ultralytics import YOLO

FRAMES_FOLDER = r"D:\F3-Project\data\frames"
CROPS_OUTPUT  = r"D:\F3-Project\data\crops"
model = YOLO("yolov8m-pose.pt")
CONF_THRESHOLD = 0.3
PADDING = 20  # pixels of padding around each person crop

os.makedirs(CROPS_OUTPUT, exist_ok=True)

total_crops = 0

worker_folders = sorted(os.listdir(FRAMES_FOLDER))

for worker in worker_folders:
    worker_frame_dir = os.path.join(FRAMES_FOLDER, worker)
    if not os.path.isdir(worker_frame_dir):
        continue

    out_dir = os.path.join(CROPS_OUTPUT, worker)
    os.makedirs(out_dir, exist_ok=True)

    frames = sorted(f for f in os.listdir(worker_frame_dir) if f.endswith(".jpg"))

    worker_crops = 0

    for frame_file in frames:
        frame_path = os.path.join(worker_frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            continue

        frame_name = os.path.splitext(frame_file)[0]
        person_idx = 0

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Add padding, clamp to image boundaries
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(w, x2 + PADDING)
            y2 = min(h, y2 + PADDING)

            crop = frame[y1:y2, x1:x2]

            # Skip tiny crops (noise)
            if crop.shape[0] < 50 or crop.shape[1] < 30:
                continue

            crop_name = f"{frame_name}_person{person_idx+1:02d}.jpg"
            cv2.imwrite(os.path.join(out_dir, crop_name), crop)
            person_idx += 1
            worker_crops += 1

    print(f"[{worker}] {worker_crops} person crops saved")
    total_crops += worker_crops

print(f"\nDone — {total_crops} total crops saved to {CROPS_OUTPUT}")