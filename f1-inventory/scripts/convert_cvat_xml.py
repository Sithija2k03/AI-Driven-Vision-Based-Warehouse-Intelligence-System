"""
convert_cvat_xml.py

Converts a CVAT "for images 1.1" XML export (with a custom occlusion_level
attribute per box) into two things:

1. Standard YOLO .txt label files (one per image), for training the
   product detector.
2. A single occlusion_labels.csv mapping each box to its occlusion level
   and pixel crop coordinates, for building the occlusion-classifier
   training set (crop each row's box out of its image using the given
   coordinates, label the crop with its occlusion_level).

Usage:
    python convert_cvat_xml.py <annotations.xml> <output_folder> [--class-id 0]

Output:
    output_folder/
        labels/
            cam08__frame_00000.txt
            cam08__frame_00003.txt
            ...
        occlusion_labels.csv

Notes:
- Class is assumed single-class ("product" -> class id 0). Change with
  --class-id if you ever add more classes.
- Images with 0 boxes still get an empty .txt file (valid YOLO behavior
  for "background" images, though you shouldn't have any here).
- Duplicate boxes: if two boxes in the same image overlap with IoU > 0.95
  (e.g. a box pasted/propagated twice), only the first is kept. The count
  removed is printed. Use --dedupe-iou 0 to disable.
- Rotated boxes: CVAT stores xtl/ytl/xbr/ybr as the UNROTATED corners plus
  a separate `rotation` angle (degrees, about the box centre). YOLO labels
  have no rotation field, so for rotated boxes this script rotates the four
  corners and writes the axis-aligned rectangle that encloses them. The
  occlusion CSV uses the same rectangle, so crops match the YOLO boxes.
"""

import sys
import csv
import math
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def convert(xml_path: Path, output_folder: Path, class_id: int = 0, dedupe_iou: float = 0.95):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels_dir = output_folder / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_folder / "occlusion_labels.csv"
    csv_rows = []

    n_images = 0
    n_boxes = 0
    n_dupes = 0
    occ_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}

    for image_el in root.findall("image"):
        n_images += 1
        img_name = image_el.get("name")
        img_w = float(image_el.get("width"))
        img_h = float(image_el.get("height"))
        stem = Path(img_name).stem

        yolo_lines = []
        kept_rects = []
        for box_el in image_el.findall("box"):
            xtl = float(box_el.get("xtl"))
            ytl = float(box_el.get("ytl"))
            xbr = float(box_el.get("xbr"))
            ybr = float(box_el.get("ybr"))

            rotation = float(box_el.get("rotation", "0") or 0)
            if rotation:
                cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
                a = math.radians(rotation)
                cos_a, sin_a = math.cos(a), math.sin(a)
                xs, ys = [], []
                for px in (xtl, xbr):
                    for py in (ytl, ybr):
                        dx, dy = px - cx, py - cy
                        xs.append(cx + dx * cos_a - dy * sin_a)
                        ys.append(cy + dx * sin_a + dy * cos_a)
                xtl, xbr, ytl, ybr = min(xs), max(xs), min(ys), max(ys)

            # clip to image bounds just in case
            xtl = max(0.0, min(xtl, img_w))
            xbr = max(0.0, min(xbr, img_w))
            ytl = max(0.0, min(ytl, img_h))
            ybr = max(0.0, min(ybr, img_h))

            # drop exact/near-exact duplicate boxes (e.g. from propagate/copy-paste)
            rect = (xtl, ytl, xbr, ybr)
            if dedupe_iou and any(iou(rect, k) > dedupe_iou for k in kept_rects):
                n_dupes += 1
                continue
            kept_rects.append(rect)
            n_boxes += 1
            box_idx = len(kept_rects) - 1

            box_w = xbr - xtl
            box_h = ybr - ytl
            x_center = xtl + box_w / 2
            y_center = ytl + box_h / 2

            # normalize to 0-1 for YOLO format
            x_center_n = x_center / img_w
            y_center_n = y_center / img_h
            w_n = box_w / img_w
            h_n = box_h / img_h

            yolo_lines.append(
                f"{class_id} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}"
            )

            # occlusion level attribute
            occ_level = "unknown"
            for attr_el in box_el.findall("attribute"):
                if attr_el.get("name") == "occlusion_level":
                    occ_level = (attr_el.text or "unknown").strip()
                    break
            occ_counts[occ_level] = occ_counts.get(occ_level, 0) + 1

            csv_rows.append({
                "image_name": img_name,
                "box_index": box_idx,
                "xtl": round(xtl, 2),
                "ytl": round(ytl, 2),
                "xbr": round(xbr, 2),
                "ybr": round(ybr, 2),
                "occlusion_level": occ_level,
            })

        label_path = labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_name", "box_index", "xtl", "ytl", "xbr", "ybr", "occlusion_level"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Converted {n_images} images, {n_boxes} boxes.")
    if dedupe_iou:
        print(f"Removed {n_dupes} duplicate boxes (IoU > {dedupe_iou} with another box in the same image).")
    print(f"YOLO labels written to: {labels_dir}")
    print(f"Occlusion CSV written to: {csv_path}")
    print(f"Occlusion level counts: {occ_counts}")
    if occ_counts.get("unknown", 0) > 0:
        print(f"WARNING: {occ_counts['unknown']} boxes had no occlusion_level attribute - check the source XML.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", help="Path to CVAT 'for images 1.1' annotations.xml")
    parser.add_argument("output_folder", help="Where labels/ and occlusion_labels.csv will be written")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id to use (default 0 = product)")
    parser.add_argument("--dedupe-iou", type=float, default=0.95,
                        help="Drop a box if it overlaps an earlier box in the same image with IoU above this "
                             "(default 0.95; use 0 to keep everything)")
    args = parser.parse_args()

    xml_path = Path(args.xml_path)
    output_folder = Path(args.output_folder)

    if not xml_path.exists():
        print(f"XML file not found: {xml_path}")
        sys.exit(1)

    convert(xml_path, output_folder, args.class_id, args.dedupe_iou)


if __name__ == "__main__":
    main()
