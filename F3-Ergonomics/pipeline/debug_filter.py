"""
debug_filter.py — inspect what confidence_filter.check_confidence actually returns
Run: py -3.12 D:\F3-Project\pipeline\debug_filter.py
"""
import sys
sys.path.insert(0, r"D:\F3-Project\pipeline")
from confidence_filter import check_confidence, reset_worker

# All good keypoints
kp_good = [[100 + i*10, 200 + i*5, 0.9] for i in range(17)]
result = check_confidence(kp_good, "debug_worker_1")

print("=== Raw return value ===")
print(repr(result))
print()
print("Type:", type(result))
if isinstance(result, (list, tuple)):
    print("Length:", len(result))
    for i, v in enumerate(result):
        print(f"  [{i}] type={type(v).__name__}  value={repr(v)}")