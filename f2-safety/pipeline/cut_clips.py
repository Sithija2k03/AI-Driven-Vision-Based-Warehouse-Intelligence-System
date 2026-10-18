"""
Step 2. Cut the chosen events into short clips (max 30 s each).

Usage:  python cut_clips.py VIDEO events.csv OUT_DIR [what_filter]
  what_filter (optional): keep only rows whose 'what' contains this text,
  e.g. forklift   or   worker   or   motion
"""
import csv, subprocess, sys
from pathlib import Path

video, csv_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
flt = sys.argv[4] if len(sys.argv) > 4 else ""
MAXLEN = 30
out_dir.mkdir(parents=True, exist_ok=True)
name = Path(video).stem

n = 0
for k, row in enumerate(csv.DictReader(open(csv_path))):
    if flt and flt not in row["what"]:
        continue
    s, e = float(row["start_s"]), float(row["end_s"])
    tag = row["what"].replace("+", "-")
    part, t = 0, s
    while t < e - 1:
        end = min(t + MAXLEN, e)
        out = out_dir / f"{name}_e{k:03d}p{part}_{tag}.mp4"
        subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-to", str(end), "-i", video,
                        "-c:v", "libx264", "-crf", "23", "-an", str(out), "-loglevel", "error"], check=True)
        n += 1; part += 1; t = end
print(f"wrote {n} clips to {out_dir}")
