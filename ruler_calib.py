import cv2
import numpy as np

ruler = cv2.imread("ruler.png", cv2.IMREAD_UNCHANGED)

if ruler is None:
    raise FileNotFoundError("Could not find ruler.png")

# Handle transparency by blending RGBA onto white background
if ruler.shape[2] == 4:
    alpha = ruler[:, :, 3] / 255.0
    bgr = ruler[:, :, :3]
    white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
    ruler_display = (
        bgr * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])
    ).astype(np.uint8)
else:
    ruler_display = ruler.copy()

# Known distance between two major ticks on your HIView ruler
KNOWN_DISTANCE_UM = 100.0  # 0.10 mm = 100 um

# Draw ROI across two major tick marks (0.10mm)
r = cv2.selectROI(
    "Select distance between two major ticks (0.10mm)", ruler_display
)
cv2.destroyAllWindows()

measured_pixels = r[2]  # Bounding box width

# Calculate TRUE pixels per micrometer for your setup
PIXELS_PER_UM = measured_pixels / KNOWN_DISTANCE_UM

# Verification against measured scale
calculated_um = measured_pixels / PIXELS_PER_UM
error_pct = abs(calculated_um - KNOWN_DISTANCE_UM) / KNOWN_DISTANCE_UM * 100.0

print("--- SCALE CALIBRATION RESULTS ---")
print(f"Measured Pixel Width: {measured_pixels} px")
print(f"Calibrated Scale Factor: {PIXELS_PER_UM:.4f} px/um")
print(f"Calculated Physical Width: {calculated_um:.2f} um")
print(f"Target Known Width: {KNOWN_DISTANCE_UM:.2f} um")
print(f"Error: {error_pct:.2f}%")
