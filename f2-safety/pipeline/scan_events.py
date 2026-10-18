"""
Step 1. Scan a long CCTV video and list the times when something happens.
Empty scenes are skipped: the detector only runs when there is motion.

Usage:  python scan_events.py VIDEO events.csv [MODEL]
Output: events.csv with start_s, end_s, what, max_conf
"""
import cv2, csv, sys, time
from ultralytics import YOLO

video = sys.argv[1]
out_csv = sys.argv[2] if len(sys.argv) > 2 else "events.csv"
model_path = sys.argv[3] if len(sys.argv) > 3 else "best.pt"

STEP_S = 1.0       # look at 1 frame per second
PAD_S = 8          # add 8 s before and after each event
GAP_S = 10         # hits closer than 10 s are one event
MOTION_T = 0.004   # share of pixels that changed; lower = more sensitive
CROP_H = 880       # keep the top part (Camera 11 has a bar at the bottom)
CONF = 0.25
IMGSZ = 960

model = YOLO(model_path)
names = model.names
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"Cannot open video: {video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
step = max(1, int(round(fps * STEP_S)))
dur = total / fps if total else 0

prev, i, hits, t0 = None, 0, [], time.time()
while True:
    if not cap.grab():
        break
    if i % step == 0:
        ok, f = cap.retrieve()
        if not ok:
            break
        f = f[:CROP_H]
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        motion = (cv2.absdiff(g, prev) > 25).mean() if prev is not None else 0.0
        prev = g
        t = i / fps
        if motion > MOTION_T:
            r = model.predict(f, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
            found = {names[int(b.cls[0])] for b in r.boxes}
            best = max([float(b.conf[0]) for b in r.boxes], default=0.0)
            hits.append((t, found, best))
        if i % (step * 600) == 0 and i > 0:
            print(f"scanned {int(t/60)} min, hits so far {len(hits)}, elapsed {int(time.time()-t0)} s", flush=True)
    i += 1
cap.release()

# merge nearby hits into events
events = []
for t, found, best in hits:
    if events and t - events[-1]["end"] <= GAP_S:
        e = events[-1]
        e["end"] = t; e["found"] |= found; e["conf"] = max(e["conf"], best)
    else:
        events.append({"start": t, "end": t, "found": set(found), "conf": best})

with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["start_s", "end_s", "what", "max_conf"])
    for e in events:
        s = max(0, e["start"] - PAD_S)
        en = e["end"] + PAD_S
        if dur:
            en = min(dur, en)
        w.writerow([round(s, 1), round(en, 1), "+".join(sorted(e["found"])) or "motion", round(e["conf"], 2)])

tot = sum(min(dur or 1e9, e["end"] + PAD_S) - max(0, e["start"] - PAD_S) for e in events)
print(f"done: {len(events)} events, about {int(tot/60)} min of {int(dur/60)} min video, wrote {out_csv}")
