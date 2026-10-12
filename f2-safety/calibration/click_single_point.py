import cv2

clicked = []
img = cv2.imread("calibration_frame.jpeg")
window_name = "Click 2 tile corners that are exactly 60cm apart (2 tiles), then press any key"

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 2:
        clicked.append((x, y))
        cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(img, str(len(clicked)), (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, img)
        print(f"Clicked pixel #{len(clicked)}: ({x}, {y})")

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, click_event)
print("Click point 1, then point 2 (should be 2 tiles / 60cm apart), then press any key.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("\nFinal points:", clicked)