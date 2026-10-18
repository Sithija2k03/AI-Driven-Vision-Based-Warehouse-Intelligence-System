r"""
Find busy moments (pickers in view) in a CCTV video and extract DENSE frames from them, for the
"temporarily blocked" study (people hiding shelves for a few seconds).

The normal extracted frames are 18 minutes apart, so they rarely catch a picker. This script:

  scan     - reads the video every --step seconds, counts people with a COCO-pretrained YOLO (no training),
             writes <out>/scan_<video>.csv and prints the busiest minutes. Frames with people are saved too.
  extract  - saves one frame every --every seconds between --start and --end (seconds or hh:mm:ss),
             resized to 640 x 640 exactly like the normal extracted frames (so the shelf outlines still fit).

File names: <prefix>__t<seconds into the video, 6 digits>.jpg   e.g. cam12__t003605.jpg

Usage (f1-inventory environment):
  python scripts\dense_frames.py scan    --video "D:\...\X5_video.mp4" --prefix cam12 --out D:\Research\dense_frames\X5_day2
  python scripts\dense_frames.py extract --video "D:\...\X5_video.mp4" --prefix cam12 --out D:\Research\dense_frames\X5_day2 ^
         --start 01:05:00 --end 01:15:00 --every 3
"""
import argparse, csv
from pathlib import Path

import cv2
import numpy as np


def to_sec(t):
    if t is None:
        return None
    t = str(t)
    if ":" in t:
        parts = [float(x) for x in t.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(t)


def hms(s):
    s = int(round(s)); return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = n / fps if n and fps else 0
    return cap, fps, dur


def frames_at(cap, times):
    """Yield (t, frame) for each requested time (seconds), seeking by milliseconds."""
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if not ok:
            break
        yield t, fr


def resize640(fr, size):
    return cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)


def scan(a):
    from ultralytics import YOLO
    det = YOLO(a.person_model)
    cap, fps, dur = open_video(a.video)
    start = to_sec(a.start) or 0.0
    end = min(to_sec(a.end) or dur, dur) if dur else (to_sec(a.end) or 0)
    if end <= start:
        raise SystemExit(f"Video length unknown or --end before --start (duration read: {dur:.0f} s). Give --end.")
    out = Path(a.out); (out / "scan_people").mkdir(parents=True, exist_ok=True)
    times = np.arange(start, end, a.step)
    print(f"Video {Path(a.video).name}: {dur / 60:.1f} min at {fps:.1f} fps. Scanning {len(times)} frames every {a.step:.0f} s ...")
    rows = []
    for i, (t, fr) in enumerate(frames_at(cap, times), 1):
        img = resize640(fr, a.size)
        r = det.predict(img, classes=[0], conf=a.person_conf, imgsz=a.size, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        area = float(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum() / (a.size * a.size)) if len(boxes) else 0.0
        rows.append({"t_sec": round(float(t), 1), "time": hms(t), "people": int(len(boxes)), "person_area_share": round(area, 4)})
        if len(boxes):
            vis = img.copy()
            for b in boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            cv2.imwrite(str(out / "scan_people" / f"{a.prefix}__t{int(t):06d}.jpg"), vis)
        if i % 100 == 0:
            print(f"  {i}/{len(times)} ({hms(t)})")
    name = Path(a.video).stem
    with open(out / f"scan_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with_people = [r for r in rows if r["people"]]
    print(f"\n{len(with_people)} of {len(rows)} scanned frames have people ({len(with_people) / max(len(rows), 1):.0%}).")
    # busiest 5-minute windows
    win = 300.0
    best = []
    for s in np.arange(start, end, 60.0):
        sub = [r for r in rows if s <= r["t_sec"] < s + win]
        if sub:
            best.append((sum(r["people"] > 0 for r in sub) / len(sub), s))
    best.sort(reverse=True)
    picked = []
    for share, s in best:
        if share <= 0 or any(abs(s - p) < win for _, p in picked):
            continue
        picked.append((share, s))
        if len(picked) == 5:
            break
    if picked:
        print("Busiest 5-minute windows (share of scanned frames with a person):")
        for share, s in picked:
            print(f"  {hms(s)} - {hms(s + win)}   {share:.0%}")
        s0 = picked[0][1]
        print(f"\nNext: python scripts\\dense_frames.py extract --video \"{a.video}\" --prefix {a.prefix} --out \"{a.out}\" "
              f"--start {hms(s0)} --end {hms(s0 + win)} --every 3")
    print(f"Saved {out / f'scan_{name}.csv'} and the frames with people in {out / 'scan_people'}")


def extract(a):
    cap, fps, dur = open_video(a.video)
    start, end = to_sec(a.start), to_sec(a.end)
    if start is None or end is None or end <= start:
        raise SystemExit("Give --start and --end (seconds or hh:mm:ss), end after start.")
    out = Path(a.out) / f"dense_{hms(start).replace(':', '')}_{hms(end).replace(':', '')}"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for t, fr in frames_at(cap, np.arange(start, end, a.every)):
        cv2.imwrite(str(out / f"{a.prefix}__t{int(t):06d}.jpg"), resize640(fr, a.size), [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"Saved {n} frames ({a.every:.0f} s apart, {hms(start)} - {hms(end)}) to {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for m in ("scan", "extract"):
        p = sub.add_parser(m)
        p.add_argument("--video", required=True)
        p.add_argument("--prefix", required=True, help="camera prefix used in your frame names, e.g. cam12")
        p.add_argument("--out", required=True)
        p.add_argument("--start", default=None, help="seconds or hh:mm:ss into the video")
        p.add_argument("--end", default=None)
        p.add_argument("--size", type=int, default=640)
        if m == "scan":
            p.add_argument("--step", type=float, default=10.0, help="seconds between scanned frames")
            p.add_argument("--person-model", default="yolov8s.pt", help="COCO-pretrained YOLO (downloads automatically)")
            p.add_argument("--person-conf", type=float, default=0.35)
        else:
            p.add_argument("--every", type=float, default=3.0, help="seconds between saved frames")
    a = ap.parse_args()
    scan(a) if a.mode == "scan" else extract(a)


if __name__ == "__main__":
    main()
