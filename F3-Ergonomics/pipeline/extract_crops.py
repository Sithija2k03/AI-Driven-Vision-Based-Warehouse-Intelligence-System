
import argparse
import sys
from pathlib import Path
import cv2

YOLO_MODEL  = r"D:\F3-Project\models\yolo11m-pose.pt"
SOURCE_DIR  = r"D:\F3-Project\data\media-pipeline"
OUT_DIR     = r"D:\F3-Project\data\worker-crops"
CONF        = 0.40          # minimum detection confidence
MIN_HEIGHT  = 60            # skip crops smaller than this (pixels) — avoids tiny detections
PADDING     = 10            # pixels of padding around each crop box

def extract_crops(source_dir, out_dir, model_path, conf_thresh, min_height, padding):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    source = Path(source_dir)
    out    = Path(out_dir)

    worker_folders = sorted(source.iterdir(), key=lambda p: p.name)
    worker_folders = [f for f in worker_folders if f.is_dir()]

    print(f"\nFound {len(worker_folders)} video folders in {source_dir}")
    print(f"Output → {out_dir}\n")

    total_crops = 0
    total_frames = 0

    for folder in worker_folders:
        wid = folder.name   # e.g. Worker-1
        img_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))

        if not img_files:
            print(f"  {wid}: no images — skipping")
            continue

        out_folder = out / wid
        out_folder.mkdir(parents=True, exist_ok=True)

        folder_crops = 0
        print(f"  {wid}: {len(img_files)} frames", end="", flush=True)

        for img_path in img_files:
            frame_stem = img_path.stem   # e.g. frame_0166
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h_img, w_img = img.shape[:2]
            total_frames += 1

            results = model(img, conf=conf_thresh, verbose=False)

            if not results or len(results[0].boxes) == 0:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()   # [x1,y1,x2,y2]
            confs = results[0].boxes.conf.cpu().numpy()

            for person_idx, (box, det_conf) in enumerate(zip(boxes, confs)):
                x1, y1, x2, y2 = box
                crop_h = y2 - y1
                if crop_h < min_height:
                    continue   # skip tiny detections (background noise)

                # Add padding, clamp to image bounds
                x1p = max(0, int(x1) - padding)
                y1p = max(0, int(y1) - padding)
                x2p = min(w_img, int(x2) + padding)
                y2p = min(h_img, int(y2) + padding)

                crop = img[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue

                # Save as frame_XXXX_person_N.jpg
                out_name = f"{frame_stem}_person_{person_idx}.jpg"
                out_path = out_folder / out_name
                cv2.imwrite(str(out_path), crop)
                folder_crops += 1

        print(f" → {folder_crops} crops saved")
        total_crops += folder_crops

    print(f"\n{'─'*50}")
    print(f"✅ Done — {total_crops} crops from {total_frames} frames across {len(worker_folders)} videos")
    print(f"   Saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract person crops from full-scene frames")
    parser.add_argument("--source", default=SOURCE_DIR,  help="Input dir with Worker-N subfolders of full frames")
    parser.add_argument("--out",    default=OUT_DIR,     help="Output dir for person crops")
    parser.add_argument("--model",  default=YOLO_MODEL,  help="YOLO model path")
    parser.add_argument("--conf",   default=CONF,        type=float, help="Detection confidence threshold (default 0.4)")
    parser.add_argument("--minH",   default=MIN_HEIGHT,  type=int,   help="Min crop height in pixels (default 60)")
    parser.add_argument("--pad",    default=PADDING,     type=int,   help="Padding around crop box (default 10)")
    args = parser.parse_args()

    extract_crops(args.source, args.out, args.model, args.conf, args.minH, args.pad)