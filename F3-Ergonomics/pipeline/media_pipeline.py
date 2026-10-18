"""
media_pipeline.py  —  F3 Media Pipeline
========================================
Extracts frames from all worker CCTV videos in D:\F3-Project\data\raw-cctv\
and saves them to D:\F3-Project\data\media-pipeline\  (separate from YOLO frames)

HOW IT WORKS:
  - Scans every subfolder under raw-cctv\ for an MP4 video file
  - Extracts exactly 1 frame per second from each video
  - Saves frames as high-quality JPEGs to media-pipeline\Worker-X\
  - Logs full metadata to pipeline_log.csv

OUTPUT:
  D:\F3-Project\data\media-pipeline\Worker-X\frame_0001.jpg ...
  D:\F3-Project\outputs\pipeline_log.csv

Usage:
  py -3.12 D:\F3-Project\pipeline\media_pipeline.py
  py -3.12 D:\F3-Project\pipeline\media_pipeline.py --fps 0.5 --max_seconds 120
"""

import os
import cv2
import csv
import argparse
from datetime import datetime, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_CCTV_DIR = r"D:\F3-Project\data\raw-cctv"
FRAMES_DIR   = r"D:\F3-Project\data\media-pipeline"
LOG_PATH     = r"D:\F3-Project\outputs\pipeline_log.csv"

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=float, default=0.5,
                    help="Frames to extract per second of video (default: 0.5)")
parser.add_argument("--quality", type=int, default=95,
                    help="JPEG quality 0-100 (default: 95)")
parser.add_argument("--max_seconds", type=int, default=120,
                    help="Max seconds to extract from each video (default: 120 = 2 minutes)")
args = parser.parse_args()

EXTRACT_FPS  = args.fps         # how many frames per second to extract
JPEG_QUALITY = args.quality
MAX_SECONDS  = args.max_seconds # stop after this many seconds per video

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_video(folder_path):
    """Find the first MP4 file inside a folder (case-insensitive)."""
    for f in os.listdir(folder_path):
        if f.lower().endswith(".mp4"):
            return os.path.join(folder_path, f)
    return None

def format_duration(seconds):
    """Convert seconds to HH:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))

# ── Main ──────────────────────────────────────────────────────────────────────
def extract_worker(worker_name, video_path, out_dir):
    """
    Extract frames from one worker's video.
    Returns a dict of metadata for the log.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return None

    video_fps      = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec   = total_frames / video_fps if video_fps > 0 else 0

    # How many source frames to skip between each saved frame
    frame_interval = max(1, int(round(video_fps / EXTRACT_FPS)))

    os.makedirs(out_dir, exist_ok=True)

    saved_count  = 0
    frame_idx    = 0
    failed_count = 0

    # Max source frames to read (based on --max_seconds)
    max_src_frames = int(video_fps * MAX_SECONDS) if MAX_SECONDS > 0 else total_frames

    print(f"  Video : {os.path.basename(video_path)}")
    print(f"  FPS   : {video_fps:.2f}  |  Duration: {format_duration(duration_sec)}  |  Resolution: {width}x{height}")
    print(f"  Limit : first {MAX_SECONDS}s  |  Extracting 1 frame every {frame_interval} source frames  ({EXTRACT_FPS} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stop after max_seconds worth of frames
        if frame_idx >= max_src_frames:
            break

        if frame_idx % frame_interval == 0:
            out_name = os.path.join(out_dir, f"frame_{saved_count+1:04d}.jpg")
            success  = cv2.imwrite(out_name, frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if success:
                saved_count += 1
            else:
                failed_count += 1

        frame_idx += 1

    cap.release()

    print(f"  Saved : {saved_count} frames  →  {out_dir}\n")

    return {
        "worker"          : worker_name,
        "video_file"      : os.path.basename(video_path),
        "resolution"      : f"{width}x{height}",
        "source_fps"      : round(video_fps, 2),
        "duration"        : format_duration(duration_sec),
        "total_src_frames": total_frames,
        "extract_rate_fps": EXTRACT_FPS,
        "max_seconds"     : MAX_SECONDS,
        "frame_interval"  : frame_interval,
        "frames_saved"    : saved_count,
        "frames_failed"   : failed_count,
        "output_folder"   : out_dir,
        "processed_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_pipeline():
    if not os.path.isdir(RAW_CCTV_DIR):
        print(f"[ERROR] raw-cctv folder not found: {RAW_CCTV_DIR}")
        return

    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Collect all worker subfolders
    worker_folders = sorted([
        d for d in os.listdir(RAW_CCTV_DIR)
        if os.path.isdir(os.path.join(RAW_CCTV_DIR, d))
    ])

    if not worker_folders:
        print("[ERROR] No worker subfolders found inside raw-cctv.")
        return

    print(f"\nF3 Media Pipeline")
    print(f"=================")
    print(f"Source  : {RAW_CCTV_DIR}")
    print(f"Output  : {FRAMES_DIR}")
    print(f"Workers : {len(worker_folders)}")
    print(f"Extract : {EXTRACT_FPS} frame(s) per second  |  Max: {MAX_SECONDS}s per video\n")

    log_rows = []
    skipped  = []

    for worker_name in worker_folders:
        worker_path = os.path.join(RAW_CCTV_DIR, worker_name)
        video_path  = find_video(worker_path)

        print(f"[{worker_name}]")

        if video_path is None:
            print(f"  [SKIP] No MP4 found in {worker_path}\n")
            skipped.append(worker_name)
            continue

        out_dir = os.path.join(FRAMES_DIR, worker_name)
        result  = extract_worker(worker_name, video_path, out_dir)

        if result:
            log_rows.append(result)

    # ── Write CSV log ─────────────────────────────────────────────────────────
    if log_rows:
        fieldnames = list(log_rows[0].keys())
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nLog saved → {LOG_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_frames = sum(r["frames_saved"] for r in log_rows)
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE")
    print(f"  Workers processed : {len(log_rows)}")
    print(f"  Workers skipped   : {len(skipped)} {skipped if skipped else ''}")
    print(f"  Total frames saved: {total_frames}")
    print(f"  Log               : {LOG_PATH}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_pipeline()