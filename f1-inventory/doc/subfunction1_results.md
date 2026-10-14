# F1 sub-function 1: per-location counting and stock status (PP1 results)

Detector: `day2_s1024` (YOLOv8s, imgsz 1024), conf 0.25, NMS IoU 0.5.
Camera: X_5, day 2 (05-09-2026). Reference (full-stock) frame: `cam12__frame_00010` (not used in training).
Locations: three rectangles on the fixed view: X5-LEFT-RACK [0-0.3], X5-FLOOR [0.3-0.7], X5-RIGHT-RACK [0.7-1.0] (x as fraction of width).
Status rule (placeholder thresholds): ratio = count / reference count; STOCK-OUT < 0.10, LOW < 0.50, else IN STOCK.

## 1. Real frames (stock unchanged)

| Frame | Location | Detected | Ref | Ratio | Annotated | Status |
|---|---|---|---|---|---|---|
| 9  | Left rack  | 99  | 100 | 0.99 | 81  | IN STOCK |
| 9  | Floor      | 95  | 115 | 0.83 | 114 | IN STOCK |
| 9  | Right rack | 92  | 95  | 0.97 | 88  | IN STOCK |
| 15 | Left rack  | 101 | 100 | 1.01 | 81  | IN STOCK |
| 15 | Floor      | 120 | 115 | 1.04 | 114 | IN STOCK |
| 15 | Right rack | 95  | 95  | 1.00 | 88  | IN STOCK |

No false alarms. Ratio varied 0.83-1.04 on unchanged stock (count noise up to about 20% per location).
Lowering NMS IoU from 0.7 to 0.5 reduced whole-frame over-counting from +13-26% to +1-12% (2 frames).

## 1b. Other cameras, real day-2 frames (one whole-frame location per camera)

| Camera | Frame | Detected | Ref | Ratio | Annotated | Count error | Status |
|---|---|---|---|---|---|---|---|
| X_5 | 9  | 286 | 310 | 0.92 | 283 | +1%  | IN STOCK |
| X_5 | 15 | 316 | 310 | 1.02 | 283 | +12% | IN STOCK |
| G_H | 7 (15:04, darker) | 126 | 154 | 0.82 | 187 | -33% | IN STOCK |
| G_H | 25 (11:28) | 148 | 154 | 0.96 | 204 | -27% | IN STOCK |
| CP2 (checkingPart_2) | 0 (08:00) | 251 | 250 | 1.00 | 241 | +4% | IN STOCK |
| CP2 (checkingPart_2) | 12 (14:00) | 256 | 250 | 1.02 | 235 | +9% | IN STOCK |

Reference frames: X_5 frame 10, G_H frame 38, CP2 frame 5 (all day 2, not used in training).
Note: checkingPart_2 day 2 was the validation split (used to pick the best epoch), so CP2 may be slightly optimistic; G_H and X_5 are test splits.

Findings
- Status was correct (IN STOCK) on all 6 real frames across 3 cameras; no false alarms.
- Absolute count error depends strongly on the camera: from -33% (G_H, under-counts) to +12% (X_5, over-counts).
  This is why status uses the ratio to the same camera's reference frame, not the raw count.
- Lighting matters: the darker afternoon G_H frame gave the lowest ratio (0.82) and the biggest miss.
  A large lighting change could push a full shelf towards LOW. Possible fixes: one reference frame per time of day,
  or brightness normalisation (future work).

## 2. Simulated removal (annotated cartons erased by inpainting, right rack, frame 9, seed 1)

| Removed | True left | True ratio | Detected | Detected ratio | Expected status | System status |
|---|---|---|---|---|---|---|
| 0%   | 88 | 1.00 | 92 | 0.97 | IN STOCK  | IN STOCK |
| 25%  | 66 | 0.75 | 86 | 0.91 | IN STOCK  | IN STOCK |
| 50%  | 44 | 0.50 | 63 | 0.66 | IN STOCK (on the line) | IN STOCK |
| 75%  | 22 | 0.25 | 34 | 0.36 | LOW       | LOW |
| 100% | 0  | 0.00 | 12 | 0.13 | STOCK-OUT | LOW |

True ratio uses the annotated full count (88); detected ratio uses the detected reference count (95).

Findings
- Detected count falls in the right order as cartons are removed (92 > 86 > 63 > 34 > 12): the counter tracks removals.
- It over-estimates what is left by about 0.1-0.16 of the reference at every level; with nothing left it still finds 12 boxes
  (on inpainting smudges and unannotated items), so a fully empty rack was reported LOW, not STOCK-OUT.
- Other locations were unaffected (left rack 98, floor 92-97).
- Status matched the expected status in 4 of 5 cases.

Limits
- Simulation: inpainting leaves smudges that a real empty shelf would not have; real empty shelves may behave differently.
- One frame, one location, one random seed.
- Thresholds are placeholders; final values should come from warehouse reorder levels and measured noise.
- Next: staged real removal at the warehouse; full-day scan of real frames for natural stock changes.
