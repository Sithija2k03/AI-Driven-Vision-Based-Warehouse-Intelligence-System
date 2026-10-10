import cv2
import os

RAW_CCTV_FOLDER  = r"D:\F3-Project\data\raw-cctv"
FRAMES_OUTPUT    = r"D:\F3-Project\data\frames"
FRAMES_PER_VIDEO = 30
START_PERCENT = 0.05
END_PERCENT   = 0.70

WORKERS = ["Worker-23", "Worker-24", "Worker-25"]

os.makedirs(FRAMES_OUTPUT, exist_ok=True)

for worker_name in WORKERS:
    direct_path    = os.path.join(RAW_CCTV_FOLDER, f"{worker_name}.mp4")
    subfolder_path = os.path.join(RAW_CCTV_FOLDER, worker_name, f"{worker_name}.mp4")
    if os.path.exists(direct_path):
        video_path = direct_path
    elif os.path.exists(subfolder_path):
        video_path = subfolder_path
    else:
        print(f"[SKIP] {worker_name} — MP4 not found")
        continue

    out_folder = os.path.join(FRAMES_OUTPUT, worker_name)
    os.makedirs(out_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(total_frames * START_PERCENT)
    end_frame   = int(total_frames * END_PERCENT)
    sample_range = end_frame - start_frame

    print(f"[{worker_name}] Total={total_frames}  Sampling frames {start_frame}–{end_frame}")

    positions = [start_frame + int(sample_range * i / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]

    saved = 0
    for idx, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            out_name = os.path.join(out_folder, f"frame_{idx+1:04d}.jpg")
            cv2.imwrite(out_name, frame)
            saved += 1
    cap.release()
    print(f"[{worker_name}] Done — {saved} frames saved")

print("\nAll done.")