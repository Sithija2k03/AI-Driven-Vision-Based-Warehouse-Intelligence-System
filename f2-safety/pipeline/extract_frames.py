"""
Step 3. Take frames from the clips. Skips frames that look the same as the
last saved frame, so you do not get hundreds of near-duplicates.

Usage:  python extract_frames.py CLIPS_DIR FRAMES_DIR [fps] [crop_h]
The first 2 seconds of every clip are skipped (see SKIP_S).
Frame names: <clip name>_<number>.jpg   (the part before the last _ is the
clip, so a later split by clip keeps frames of one event together)
"""
import cv2, sys
from pathlib import Path

clips, out = Path(sys.argv[1]), Path(sys.argv[2])
fps_out = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
crop_h = int(sys.argv[4]) if len(sys.argv) > 4 else 880
DIFF_T = 0.004      # skip a frame if fewer than this share of pixels changed
SKIP_S = 2.0        # skip the first 2 s of each clip (HEVC cuts can start with damaged frames)
out.mkdir(parents=True, exist_ok=True)

total = 0
for mp4 in sorted(clips.glob("*.mp4")):
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps / fps_out)))
    i, kept, last = 0, 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0 and i >= fps * SKIP_S:
            f = f[:crop_h]
            g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if last is None or (cv2.absdiff(g, last) > 25).mean() > DIFF_T:
                cv2.imwrite(str(out / f"{mp4.stem}_{kept:04d}.jpg"), f)
                last = g; kept += 1
        i += 1
    cap.release()
    total += kept
    print(f"{mp4.name}: {kept} frames")
print(f"total {total} frames in {out}")