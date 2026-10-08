import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("image.jpg")

if image is None:
    print("Error: Could not read image.")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------- Median Filter ----------------
median = cv2.medianBlur(image_rgb, 5)

# ---------------- Max Filter ----------------
kernel = np.ones((5, 5), np.uint8)
max_filter = cv2.dilate(image_rgb, kernel)

# ---------------- Min Filter ----------------
min_filter = cv2.erode(image_rgb, kernel)

# ---------------- Display ----------------
plt.figure(figsize=(12, 8))

titles = [
    "Original Image",
    "Median Filter (5x5)",
    "Max Filter (5x5)",
    "Min Filter (5x5)",
]

images = [image_rgb, median, max_filter, min_filter]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
