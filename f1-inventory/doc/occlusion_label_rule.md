# Occlusion labelling rule (F1)

Question to ask for each carton: how much of the carton's front face is hidden
(by other cartons, the rack, people, or the frame edge)?

| Level  | Hidden share | Rule in one sentence |
|--------|--------------|----------------------|
| low    | under ~30%   | The whole front face is visible, or only a small corner is covered. |
| medium | ~30% to ~70% | Part of the front face is covered, but you can still see where the carton starts and ends. |
| high   | over ~70%    | Only a strip, edge or corner of the carton is visible. |

Tie-breaks
- If unsure between two levels, choose the one closer to what is clearly visible.
- Judge the carton as it would look in full; don't count dark or blurry parts as hidden.

Used for: agreement check round 2 (outputs/agreement_round2), 16 Oct 2026.
PP2: add 2-3 example images per level.