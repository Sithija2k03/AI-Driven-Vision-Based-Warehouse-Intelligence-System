# F1 new idea: temporary occlusion (pickers) and reliability-weighted temporal fusion (PP1)

## Problem
The occlusion model handles PERMANENT occlusion (cartons hidden behind cartons). Pickers standing in front of a shelf
cause TEMPORARY occlusion: for a few seconds the count drops and a single-frame system can report a false LOW /
STOCK-OUT. Regular frames are 18 minutes apart, so this was invisible in the original evaluation.

## Method (`scripts/temporal_blocking.py`, frames from `scripts/dense_frames.py`)
1. People detection with a COCO-pretrained YOLOv8s (class person, no training). Per shelf area:
   blocked_share = share of the shelf polygon covered by (slightly enlarged) person boxes.
   blocked_share >= 0.10 -> the shelf is TEMPORARILY BLOCKED in that frame and the frame is not used for its status.
2. Reliability-weighted temporal fusion: weighted median of count/reference over the last 5 usable frames;
   weight = 1 - combined unreliability (detector confidence + occlusion trust + position; same signals as CHECK BY EYE).
3. Hysteresis: a new status is reported only after it holds for 3 usable frames.

Frozen settings (chosen on G-H, then tested unchanged on X-5): person confidence 0.15, person image size 1024,
person box enlarged by 15%, blocked threshold 0.10, window 5, persist 3, LOW < 0.5, STOCK-OUT < 0.1.

## Data
Dense frames, one every 3 s, from day-2 videos (never used for training):
- Development: G-H (cam10), 08:20-08:25 and 11:10-11:20 -> 300 frames x 9 shelf areas = 2,700 shelf readings.
  Busy camera: a person is visible in 39% of scanned moments.
- Fair test: X-5 (cam12), 11:54-12:04 and 10:33-10:38 -> 300 frames x 4 shelf areas = 1,200 shelf readings.
  Quieter camera: a person is visible in 14% of scanned moments.

## Evaluation without extra labels
Inside a 5-10 minute window the real stock of a shelf barely changes. Reference status per shelf and window =
status of the median ratio over frames with no person on that shelf. Measured: share of readings whose status differs
from the reference, false LOW/STOCK-OUT alarms, status flips inside the window, ratio variation (coefficient of variation).

## Results

### Development: G-H (2,700 shelf readings)
v1 (person confidence 0.35, image 640) missed pickers seen from above at the top edge of the frame (9 "unexplained drops",
confirmed visually). v2 settings were chosen to reduce missed pickers (1 unexplained drop), not on the final scores.

| G-H, v2 settings | Single-frame (current system) | Proposed |
|---|---|---|
| Readings blocked by a picker | 17% | |
| Status differs from reference | 8.0% | 6.1% |
| False LOW / STOCK-OUT alarms | 217 | 67 |
| Status flips inside windows | 61 | 6 |
| Ratio variation (mean CV) | 0.075 | 0.035 |
| Single-frame error rate, blocked vs clear readings | 37.7% vs 2.1% | |

- Pickers cause the errors: when a picker blocks a shelf the single-frame status is wrong 38% of the time vs 2% otherwise.
- 17 of 18 shelf-windows: single-frame 51 false alarms -> proposed 0.
- All remaining 67 false alarms come from one shelf-window (V1-V3, 11:10-11:20), a picking hotspot on the upper
  level where a picker is in front of the shelf in 66% of frames and often only partly visible.

### Fair test: X-5, settings unchanged (1,200 shelf readings)
| X-5 | Single-frame | Proposed |
|---|---|---|
| Readings blocked by a picker | 4% | |
| Status differs from reference | 0.0% | 0.0% |
| False LOW / STOCK-OUT alarms | 0 | 0 |
| Status flips | 0 | 0 |
| Ratio variation (mean CV) | 0.045 | 0.034 |
| Unexplained drops (missed pickers) | 0 | |

- On X-5 pickers rarely block shelves (4% of readings) and never changed the single-frame status, so there were no
  errors to remove. The proposed method introduced no new errors and made the count 24% steadier.

## Findings
1. Temporary occlusion by pickers is a real, measurable error source on busy cameras (G-H: 38% vs 2% status error).
2. Detecting pickers + reliability-weighted fusion + hysteresis removed almost all picker-caused false alarms on G-H
   and cut status flicker by about 90%, while keeping status changes for persistent drops (one change instead of 12 flips).
3. On a quieter camera (X-5) the method is safe: no added errors, steadier counts.

## Limitations
- Development and test cameras differ in how busy they are; the error-reduction result comes from G-H only.
- Shelves that are blocked most of the time (V1-V3) cannot be confirmed from video; they should be reported as
  "busy area, not confirmed" (planned, not implemented/tuned on this data).
- People seen from above at the frame edge are still sometimes missed; the reference status assumes stock is
  constant over 5-10 minutes; 3-frame confirmation adds about 9-30 s delay to real status changes.
- One day, two cameras, about 3,900 shelf readings.

## Next (PP2)
"Busy area" status for hotspot shelves; a people detector fine-tuned on top-down warehouse views; more cameras and days;
live streaming with the TEMPORARILY BLOCKED status sent to F4.
