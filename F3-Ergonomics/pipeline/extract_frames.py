"""
extract_frames.py
Extracts 15 evenly-spaced frames from each of the 20 worker videos.
Handles both:
  - Direct MP4 files: D:\F3-Project\data\raw-cctv\Worker-7.mp4
  - Subfolder MP4s:   D:\F3-Project\data\raw-cctv\Worker-1\Worker-1.mp4
Output: D:\F3-Project\data\frames\Worker-X\frame_0001.jpg
"""

import cv2
import os

# ── Configuration ──────────────────────────────────────────────────────────
RAW_CCTV_FOLDER  = r"D:\F3-Project\data\raw-cctv"
FRAMES_OUTPUT    = r"D:\F3-Project\data\frames"
FRAMES_PER_VIDEO = 15   # 15 frames per worker

# Extract from middle 60% of video (skip first 20% and last 20%)
# This avoids camera setup at start and end where workers may not be visible
FRAMES_PER_VIDEO = 30
START_PERCENT = 0.05
END_PERCENT   = 0.70


# ── All 20 workers ─────────────────────────────────────────────────────────
WORKERS = [
    "Worker-1",  "Worker-2",  "Worker-3",  "Worker-4",
    "Worker-5",  "Worker-6",  "Worker-7",  "Worker-8",
    "Worker-9",  "Worker-10", "Worker-11", "Worker-12",
    "Worker-13", "Worker-14", "Worker-15", "Worker-16",
    "Worker-17", "Worker-18", "Worker-19", "Worker-20",
]

# ── Frame extraction ───────────────────────────────────────────────────────
os.makedirs(FRAMES_OUTPUT, exist_ok=True)

for worker_name in WORKERS:

    # Find the MP4 — either direct file or inside subfolder
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

    # Only sample from the middle 60% of the video
    start_frame = int(total_frames * START_PERCENT)
    end_frame   = int(total_frames * END_PERCENT)
    sample_range = end_frame - start_frame

    print(f"[{worker_name}] Total={total_frames}  Sampling frames {start_frame}–{end_frame}")

    # Pick evenly spaced positions within that range
    positions = [
        start_frame + int(sample_range * i / FRAMES_PER_VIDEO)
        for i in range(FRAMES_PER_VIDEO)
    ]

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

print("\nAll 20 workers done.")
