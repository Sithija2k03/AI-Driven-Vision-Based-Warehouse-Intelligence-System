import cv2
import numpy as np

# Load the homography matrix you already saved
H = np.load("homography_matrix.npy")

def pixel_to_realworld(pixel_x, pixel_y, H):
    point = np.array([pixel_x, pixel_y, 1.0], dtype=np.float32)
    mapped = H @ point
    mapped /= mapped[2]
    return mapped[0], mapped[1]

# Your two clicked tile-corner pixels
point1_px = (541, 320)
point2_px = (801, 303)

# Convert both to real-world metres
x1, y1 = pixel_to_realworld(point1_px[0], point1_px[1], H)
x2, y2 = pixel_to_realworld(point2_px[0], point2_px[1], H)

print(f"Point 1 -> real-world: ({x1:.3f}, {y1:.3f}) m")
print(f"Point 2 -> real-world: ({x2:.3f}, {y2:.3f}) m")

# Compute distance between them
computed_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
known_distance = 0.60  # 2 tiles x 30cm

error = abs(computed_distance - known_distance)
error_percent = (error / known_distance) * 100

print(f"\nComputed distance: {computed_distance:.3f} m")
print(f"Known distance (ground truth): {known_distance:.3f} m")
print(f"Error: {error*100:.2f} cm  ({error_percent:.1f}%)")