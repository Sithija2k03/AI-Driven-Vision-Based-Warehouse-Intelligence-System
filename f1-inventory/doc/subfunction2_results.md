# F1 sub-function 2: occlusion-aware compliance (PP1 results)

## Step 1. Detector recall by occlusion level

Question: does the detector miss more cartons when they are more hidden?
Method: `scripts/recall_by_occlusion.py`. Detector `day2_s1024` (conf 0.25, NMS IoU 0.5, imgsz 1024).
Each annotated carton counts as found if one predicted box overlaps it with IoU >= 0.5 (one-to-one, greedy).
Data: the three day-2 batches, never used for training (37 frames, 8,846 cartons).

| Batch | Low | Medium | High |
|---|---|---|---|
| G_H day 2 (test) | 0.705 (340/482) | 0.545 (515/945) | 0.338 (350/1034) |
| X_5 day 2 (test) | 0.667 (340/510) | 0.542 (906/1673) | 0.412 (475/1152) |
| checkingPart_2 day 2 (val) | 0.572 (170/297) | 0.625 (899/1438) | 0.475 (624/1315) |
| **All** | **0.659** (850/1289) | **0.572** (2320/4056) | **0.414** (1449/3501) |

Findings
- Overall recall falls from 0.66 (low) to 0.57 (medium) to 0.41 (high): about 59% of highly occluded cartons are missed,
  against about 34% of low-occlusion cartons.
- High occlusion has the lowest recall in every batch. Low > medium > high holds in G_H and X_5;
  in checkingPart_2 low is slightly below medium (only 297 low boxes; the intervals overlap).
- Conclusion: carton counts are least reliable where occlusion is high, which supports estimating occlusion per box
  and using it to judge how far a count can be trusted.

Limits
- 95% intervals (Wilson) treat cartons as independent; frames in a batch are near-duplicates, so true uncertainty is larger.
- Occlusion levels come from one annotator (agreement check pending).
- checkingPart_2 day 2 was the validation split.

### Robustness check: looser match (IoU >= 0.3)

| Batch | Low | Medium | High |
|---|---|---|---|
| G_H day 2 | 0.772 | 0.619 | 0.473 |
| X_5 day 2 | 0.745 | 0.735 | 0.569 |
| checkingPart_2 day 2 | 0.818 | 0.778 | 0.703 |
| **All** | **0.772** (995/1289) | **0.723** (2933/4056) | **0.591** (2069/3501) |

- With a looser match, low > medium > high now holds in all three batches.
- The low-to-high gap shrinks from 0.245 (IoU 0.5) to 0.181 (IoU 0.3): part of the drop is loose boxing of
  partly hidden cartons, but most of it remains, so highly occluded cartons are genuinely missed more often.
- About 41% of highly occluded cartons are not found even with the looser match (against about 23% of low ones).
- Caveat: at IoU 0.3 in dense scenes a prediction can occasionally match a neighbouring carton, so these values are
  an upper bound on recall.
