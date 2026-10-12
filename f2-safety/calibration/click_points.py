import cv2

clicked_points = []
img = cv2.imread("calibration_frame.jpeg")

if img is None:
    print("ERROR: Image not found. Check the filename and that you're in the 'calibration' folder.")
    exit()

window_name = "Click the 4 floor points IN ORDER (A, B, C, D)"

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= 4:
            print("Already have 4 points. Press any key to finish, or 'r' to restart.")
            return
        clicked_points.append((x, y))
        # draw a visible marker + number at the clicked point
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img, str(len(clicked_points)), (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow(window_name, img)
        print(f"Clicked pixel #{len(clicked_points)}: ({x}, {y})")

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, click_event)

print("Click your 4 points in order (A, B, C, D). Press any key once all 4 are marked.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nFinal pixel points, in click order:")
print(clicked_points)

if len(clicked_points) != 4:
    print(f"\nWARNING: You clicked {len(clicked_points)} points, not 4. Please re-run the script.")