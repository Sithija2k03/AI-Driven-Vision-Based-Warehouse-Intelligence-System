r"""
Draw shelf outlines (polygons) on a camera frame and save them as locations for count_stock.py.

Draw once per camera, on its full-stock reference frame. Every later frame from the same
camera uses the same outlines (as long as the camera does not move).

Usage (from the f1-inventory folder):
  python scripts\draw_shelves.py --camera-id G_H --config configs\shelves.json ^
      --frame "D:\Research\research camera data\05-09-2026\_LOCATION__G___H_DVR1_STORE_20260918110921\extracted_frames\cam10__frame_00038.jpg"

Controls (click on the image window first so it has focus):
  left click      add a corner point
  right click / u undo the last point
  n or Enter      finish this shelf -> type its name (e.g. G-22) in the TERMINAL, press Enter,
                  then click back on the image window
  d               delete the last finished shelf
  x               delete shelves by name (type names in the TERMINAL, e.g. G-23, G-24)
  s               save and quit
  q or Esc        quit WITHOUT saving

Tips:
  - Go around each shelf's corners in order (clockwise or anti-clockwise), 4-6 points is enough.
  - Use the exact names the warehouse uses (same as F4's layout), e.g. G-22, V4, A.
  - Try not to let shelves overlap. If they do, count_stock.py gives a box to the smaller shelf.
  - Running again for the same camera loads the saved shelves so you can add or fix some.
"""
import argparse, json, shutil
from pathlib import Path

import cv2
import numpy as np

BAR = 24  # height of the help bar under the image
COLOURS = [(0, 255, 255), (255, 0, 255), (0, 200, 0), (255, 128, 0), (0, 128, 255),
           (255, 255, 0), (128, 0, 255), (0, 0, 255), (200, 200, 200), (0, 255, 128)]


def load(config_path, camera_id):
    cfg = json.load(open(config_path, encoding="utf-8")) if Path(config_path).exists() else {"cameras": {}}
    cfg.setdefault("cameras", {})
    cam = cfg["cameras"].get(camera_id, {})
    shelves = [{"location_id": l["location_id"], "polygon": [list(p) for p in l["polygon"]]}
               for l in cam.get("locations", []) if "polygon" in l]
    return cfg, shelves


def save(config_path, cfg, camera_id, frame, shelves):
    p = Path(config_path)
    if p.exists():
        shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
    cam = cfg["cameras"].get(camera_id, {})
    cam["reference_frame"] = str(frame)
    old = {l["location_id"]: l for l in cam.get("locations", [])}
    locs = []
    for s in shelves:
        loc = {k: v for k, v in old.get(s["location_id"], {}).items() if k not in ("region", "polygon")}
        loc["location_id"] = s["location_id"]
        loc.setdefault("slots", [])   # fill with F4 slot IDs, e.g. ["G-19", "G-20"]
        loc["polygon"] = [[round(x, 4), round(y, 4)] for x, y in s["polygon"]]
        locs.append(loc)
    cam["locations"] = locs
    cfg["cameras"][camera_id] = cam
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2)


def render(base, shelves, current, scale, show_help=True):
    img = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    over = img.copy()
    for i, s in enumerate(shelves):
        pts = np.array([[int(x * w), int(y * h)] for x, y in s["polygon"]], np.int32)
        c = COLOURS[i % len(COLOURS)]
        cv2.fillPoly(over, [pts], c)
        cv2.polylines(img, [pts], True, c, 2)
    img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
    for i, s in enumerate(shelves):
        pts = np.array([[int(x * w), int(y * h)] for x, y in s["polygon"]], np.int32)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        (tw, th), _ = cv2.getTextSize(s["location_id"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (cx - tw // 2 - 3, cy - th - 4), (cx + tw // 2 + 3, cy + 4), (0, 0, 0), -1)
        cv2.putText(img, s["location_id"], (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLOURS[i % len(COLOURS)], 2, cv2.LINE_AA)
    if current:
        pts = np.array([[int(x * w), int(y * h)] for x, y in current], np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 2)
        for p in pts:
            cv2.circle(img, tuple(int(v) for v in p), 4, (255, 255, 255), -1)
    if not show_help:
        return img
    help_txt = "click=point u=undo n=finish d=del last x=del by name s=save q=quit"
    bar = np.zeros((BAR, w, 3), np.uint8)   # help bar BELOW the frame, so it hides nothing
    cv2.putText(bar, help_txt, (6, BAR - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="a full-stock frame of this camera (it becomes the reference frame)")
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--config", default="configs/shelves.json")
    ap.add_argument("--scale", type=float, default=1.5, help="window zoom (640 px frame -> 960 px window)")
    ap.add_argument("--fresh", action="store_true", help="ignore shelves already saved for this camera")
    a = ap.parse_args()

    base = cv2.imread(a.frame)
    if base is None:
        raise SystemExit(f"Could not read {a.frame}")
    cfg, shelves = load(a.config, a.camera_id)
    if a.fresh:
        shelves = []
    print(f"Loaded {len(shelves)} saved shelves for {a.camera_id}." if shelves else "Starting with no shelves.")
    current = []
    win = f"Draw shelves - {a.camera_id}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    size = {}

    def on_mouse(event, x, y, flags, param):
        if not size:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((min(max(x / size["w"], 0.0), 1.0), min(max(y / size["h"], 0.0), 1.0)))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = render(base, shelves, current, a.scale)
        size["h"], size["w"] = view.shape[0] - BAR, view.shape[1]
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed - nothing saved."); break
            continue
        if k in (ord("u"), 8) and current:
            current.pop()
        elif k in (ord("n"), 13):
            if len(current) < 3:
                print("A shelf needs at least 3 points."); continue
            name = input("Name for this shelf (e.g. G-22), empty = cancel: ").strip()
            if not name:
                print("Cancelled - points kept, keep clicking or press u to undo."); continue
            if any(s["location_id"] == name for s in shelves):
                print(f"'{name}' already exists - replacing it.")
                shelves = [s for s in shelves if s["location_id"] != name]
            shelves.append({"location_id": name, "polygon": [list(p) for p in current]})
            current = []
            print(f"Added {name}. Shelves so far: {', '.join(s['location_id'] for s in shelves)}")
            print("Click back on the image window to continue.")
        elif k == ord("d") and shelves:
            print(f"Deleted {shelves.pop()['location_id']}")
        elif k == ord("x") and shelves:
            names = input(f"Delete which shelves? ({', '.join(s['location_id'] for s in shelves)})\n"
                          "Type names separated by commas, empty = cancel: ").strip()
            for n in [n.strip() for n in names.split(",") if n.strip()]:
                if any(s["location_id"] == n for s in shelves):
                    shelves = [s for s in shelves if s["location_id"] != n]
                    print(f"Deleted {n}")
                else:
                    print(f"No shelf called '{n}'")
            print("Click back on the image window to continue.")
        elif k == ord("s"):
            if current:
                print("You have an unfinished shelf - press n to finish it or u to remove its points first."); continue
            save(a.config, cfg, a.camera_id, a.frame, shelves)
            print(f"Saved {len(shelves)} shelves for {a.camera_id} to {a.config}")
            cv2.imwrite(str(Path(a.config).with_name(f"shelves_{a.camera_id}.jpg")), render(base, shelves, [], 1.0, show_help=False))
            print(f"Picture of the outlines: {Path(a.config).with_name(f'shelves_{a.camera_id}.jpg')}")
            break
        elif k in (ord("q"), 27):
            print("Quit - nothing saved."); break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
