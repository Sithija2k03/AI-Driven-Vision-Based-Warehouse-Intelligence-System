import cv2
import numpy as np

# Real-world coordinates (metres), matching the SAME order you clicked: A, B, C, D
real_world_points = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [2.0, 1.2],
    [0.0, 1.2]
], dtype=np.float32)

# Pixel coordinates from click_points.py, same order
pixel_points = np.array([
    [151, 123],
    [954, 86],
    [1078, 535],
    [91, 615]
], dtype=np.float32)

# Compute the homography matrix
H, status = cv2.findHomography(pixel_points, real_world_points)
print("Homography matrix:")
print(H)

# Save it for later use in the distance/TTC pipeline
np.save("homography_matrix.npy", H)
print("\nSaved to homography_matrix.npy")


def pixel_to_realworld(pixel_x, pixel_y, H):
    """Convert one pixel coordinate to real-world (x, y) in metres."""
    point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float32)
    mapped = H @ point
    mapped /= mapped[2]
    return mapped[0], mapped[1]


# Sanity check: converting point A back should give ~(0.0, 0.0)
test_x, test_y = pixel_to_realworld(151, 123, H)
print(f"\nSanity check A -> should be close to (0.0, 0.0): ({test_x:.3f}, {test_y:.3f})")

# Sanity check: converting point C back should give ~(2.0, 1.2)
test_x2, test_y2 = pixel_to_realworld(1078, 535, H)
print(f"Sanity check C -> should be close to (2.0, 1.2): ({test_x2:.3f}, {test_y2:.3f})")
