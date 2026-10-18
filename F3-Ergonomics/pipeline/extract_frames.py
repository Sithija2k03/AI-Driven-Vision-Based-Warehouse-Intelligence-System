

import os
import cv2
import argparse

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_CCTV_BASE  = r"D:\F3-Project\data\raw-cctv"
PIPELINE_BASE  = r"D:\F3-Project\data\media-pipeline"
INTERVAL_SEC   = 1.0    # extract one frame every N seconds
MAX_FRAMES     = 80     # max frames per worker
IMG_QUALITY    = 95     # JPEG quality (0-100)
# ─────────────────────────────────────────────────────────────────────────────

def extract_worker(worker_name, video_path, out_dir, interval_sec, max_frames):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return 0

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_f / fps
    step     = max(1, int(fps * interval_sec))   # frames to skip between captures

    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  FPS={fps:.1f}  Duration={duration:.1f}s  "
          f"Total frames={total_f}  Step={step} frames")

    saved   = 0
    f_idx   = 0

    while saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(out_dir, f"frame_{saved+1:04d}.jpg")

        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, IMG_QUALITY])

        saved += 1
        f_idx += step

        if f_idx >= total_f:
            # Video shorter than expected — spread remaining extractions
            print(f"  [NOTE] Video ended at frame {f_idx}, "
                  f"only {saved} frames could be extracted.")
            break

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from raw CCTV videos")
    parser.add_argument("--interval",   type=float, default=INTERVAL_SEC,
                        help=f"Seconds between frames (default {INTERVAL_SEC})")
    parser.add_argument("--max_frames", type=int,   default=MAX_FRAMES,
                        help=f"Max frames per worker (default {MAX_FRAMES})")
    parser.add_argument("--workers",    type=str,   default=None,
                        help="Comma-separated worker names to process, e.g. Worker-1,Worker-3 "
                             "(default: all)")
    args = parser.parse_args()

    # Collect worker folders
    all_workers = sorted(
        d for d in os.listdir(RAW_CCTV_BASE)
        if os.path.isdir(os.path.join(RAW_CCTV_BASE, d))
    )

    if args.workers:
        target = [w.strip() for w in args.workers.split(",")]
        all_workers = [w for w in all_workers if w in target]

    print(f"\n{'='*60}")
    print(f"  extract_frames.py")
    print(f"  Source : {RAW_CCTV_BASE}")
    print(f"  Output : {PIPELINE_BASE}")
    print(f"  Interval: {args.interval}s  |  Max per worker: {args.max_frames}")
    print(f"  Workers : {len(all_workers)} found")
    print(f"{'='*60}\n")

    total_saved = 0

    for worker_name in all_workers:
        worker_raw_dir = os.path.join(RAW_CCTV_BASE, worker_name)
        out_dir        = os.path.join(PIPELINE_BASE,  worker_name)

        # Find the video file (same name as folder)
        video_path = os.path.join(worker_raw_dir, f"{worker_name}.mp4")
        if not os.path.exists(video_path):
            # Try .avi fallback
            video_path = os.path.join(worker_raw_dir, f"{worker_name}.avi")
        if not os.path.exists(video_path):
            # Try any mp4 in the folder
            mp4s = [f for f in os.listdir(worker_raw_dir)
                    if f.lower().endswith((".mp4", ".avi", ".mov"))]
            if mp4s:
                video_path = os.path.join(worker_raw_dir, mp4s[0])
            else:
                print(f"\n[SKIP] {worker_name} — no video file found in {worker_raw_dir}")
                continue

        print(f"\n[{worker_name}]  →  {out_dir}")
        n = extract_worker(worker_name, video_path, out_dir,
                           args.interval, args.max_frames)
        print(f"  ✓ Saved {n} frames")
        total_saved += n

    print(f"\n{'='*60}")
    print(f"  Done.  Total frames saved: {total_saved}")
    print(f"  Output folder: {PIPELINE_BASE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()