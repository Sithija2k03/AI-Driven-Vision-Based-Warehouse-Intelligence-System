import cv2
import numpy as np

# 1. Real-world coordinates (meters): Width = 3.0m, Height = 5.65m
# Order: Bottom-Left (A), Bottom-Right (B), Top-Right (C), Top-Left (D) 
# [පරීක්ෂා කරන්න: ඔබේ A, B, C, D අනුපිළිවෙළ මීට අනුකූල විය යුතුය]
real_world_points = np.array([
    [0.0, 0.0],    # Point A (Bottom-Left)
    [3.0, 0.0],    # Point B (Bottom-Right)
    [3.0, 5.65],   # Point C (Top-Right)
    [0.0, 5.65]    # Point D (Top-Left)
], dtype=np.float32)

# 2. ඔබ click_points.py මඟින් ලබාගත් සැබෑ පික්සල් අගයන් 4
pixel_points = np.array([
    [287, 657],    # Point A
    [385, 603],    # Point B
    [673, 732],    # Point C
    [595, 826]     # Point D
], dtype=np.float32)

# 3. Compute the homography matrix
H, status = cv2.findHomography(pixel_points, real_world_points)
print("Homography matrix:")
print(H)

# 4. Save it inside the calibration folder for later use
np.save("calibration/homography_matrix.npy", H)
print("\nSaved successfully to 'calibration/homography_matrix.npy'")

def pixel_to_realworld(pixel_x, pixel_y, H):
    """Convert one pixel coordinate to real-world (x, y) in metres."""
    point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float32)
    mapped = H @ point
    mapped /= mapped[2]
    return mapped[0], mapped[1]

# Sanity check: converting point A back should give ~(0.0, 0.0)
test_x, test_y = pixel_to_realworld(287, 657, H)
print(f"\nSanity check A -> should be close to (0.0, 0.0): ({test_x:.3f}, {test_y:.3f})")

# Sanity check: converting point C back should give ~(3.0, 5.65)
test_x2, test_y2 = pixel_to_realworld(673, 732, H)
print(f"Sanity check C -> should be close to (3.0, 5.65): ({test_x2:.3f}, {test_y2:.3f})")