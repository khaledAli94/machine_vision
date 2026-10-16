import cv2
import numpy as np

# --- CALIBRATION CONFIGURATION ---
PIXELS_PER_UM = 1.0700  # Derived from ruler calibration (107 px = 100 um)
VIDEO_FILE = "file.mp4"


def nothing(x):
    pass


# Window & Trackbars setup
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 450, 350)

# Detection & Filter Controls
cv2.createTrackbar("Block Size", "Controls", 5, 99, nothing)
cv2.createTrackbar("C Constant", "Controls", 48, 100, nothing)  # 48 - 50 = -2
cv2.createTrackbar("Kernel Size", "Controls", 0, 15, nothing)  # 0 = disabled
cv2.createTrackbar("Min Area (px)", "Controls", 0, 10, nothing)
cv2.createTrackbar("Max Area (px)", "Controls", 1, 15, nothing)

cap = cv2.VideoCapture(VIDEO_FILE)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Fetch trackbar positions
    block_size = cv2.getTrackbarPos("Block Size", "Controls")
    c_val = cv2.getTrackbarPos("C Constant", "Controls") - 50
    k_size = cv2.getTrackbarPos("Kernel Size", "Controls")
    min_area = cv2.getTrackbarPos("Min Area (px)", "Controls")
    max_area = cv2.getTrackbarPos("Max Area (px)", "Controls")

    # Enforce OpenCV adaptive threshold constraints
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    # Pre-processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_val,
    )

    # Optional Morphological Opening
    if k_size > 0:
        kernel = np.ones((k_size, k_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Contour Extraction
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    out = frame.copy()
    count = 0

    for c in contours:
        area_px = cv2.contourArea(c)

        # Filter by Pixel Area
        if area_px < min_area or area_px > max_area:
            continue

        # Convert pixel metrics to physical units (micrometers)
        area_um2 = area_px / (PIXELS_PER_UM**2)
        dia_um = 2 * np.sqrt(area_um2 / np.pi)

        (x, y), radius = cv2.minEnclosingCircle(c)

        # Highlight detected particle
        cv2.circle(
            out,
            (int(x), int(y)),
            max(1, int(radius)),
            (0, 255, 0),
            1,
        )

        count += 1

    # Real-time On-Screen Display
    cv2.putText(
        out,
        f"Count: {count} | Scale: {PIXELS_PER_UM:.2f} px/um",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Original", frame)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Detected", out)

    if cv2.waitKey(30) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

# BS= 5, C= 48, KS= 0 MinA=0 MaxA=30
# BS= 6, C= 47, KS= 1 MinA=0 MaxA=2
