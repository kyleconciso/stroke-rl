import cv2
import numpy as np

# Global state variables
thickness = 3
drawing = False
start_pos = None

# Create a white canvas
canvas_width, canvas_height = 512, 512
canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

def mouse_callback(event, x, y, flags, param):
    global drawing, start_pos, thickness, canvas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pos = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_pos = (x, y)
        # Draw the line using OpenCV's fast line drawing function
        cv2.line(canvas, start_pos, end_pos, color=(0, 0, 0), thickness=thickness)
    
    elif event == cv2.EVENT_MOUSEWHEEL:
        # The sign of flags usually indicates wheel direction:
        #   positive: wheel scrolled up, negative: wheel scrolled down.
        if flags > 0:
            thickness += 1
        else:
            thickness = max(1, thickness - 1)

# Create a window and set the mouse callback function
cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", mouse_callback)

while True:
    # Create a copy to display current thickness without modifying the canvas permanently
    display_img = canvas.copy()
    cv2.putText(display_img, f"Thickness: {thickness}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Canvas", display_img)
    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # Exit if ESC key is pressed
        break

cv2.destroyAllWindows()
