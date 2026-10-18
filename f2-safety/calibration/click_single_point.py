import cv2
import numpy as np

img_path = "D:\\F2_project\\calibration\\vlcsnap-2026-10-17-12h01m24s317.png"
original_img = cv2.imread(img_path)

if original_img is None:
    print("Error: Image not found! Please check the file path.")
    exit()

img = original_img.copy()
clicked = []
window_name = "Click 2 points. Press 'r' to Reset, or any key when done."

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

def click_event(event, x, y, flags, params):
    global img, clicked
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 2:
        clicked.append((x, y))
        cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(img, str(len(clicked)), (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, img)
        print(f"Clicked pixel #{len(clicked)}: ({x}, {y})")

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, click_event)
print("Click point 1, then point 2. Press 'r' to reset if you make a mistake.")

while True:
    key = cv2.waitKey(1) & 0xFF
    # 
    if key == ord('r') or key == ord('R'):
        img = original_img.copy()
        clicked = []
        cv2.imshow(window_name, img)
        print("\n[Reset] Points cleared! Click point 1 again.")
    
    # 
    elif len(clicked) == 2:
        print("\n2 points captured successfully! Press any key to close.")
        cv2.waitKey(0)
        break
    elif key != 255 and key != 255: 
        break

cv2.destroyAllWindows()
print("\nFinal points:", clicked)