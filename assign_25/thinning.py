import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_2.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("img_1.jpeg not found")

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Thinning using Zhang-Suen algorithm
thinning = cv2.ximgproc.thinning(binary)

# Skeletonization using morphological operations
skeleton = np.zeros_like(binary)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
work = binary.copy()

while cv2.countNonZero(work) > 0:
    eroded = cv2.erode(work, element)
    opened = cv2.dilate(eroded, element)
    residue = cv2.subtract(work, opened)
    skeleton = cv2.bitwise_or(skeleton, residue)
    work = eroded

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(thinning, cmap="gray")
plt.title("Thinning")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(skeleton, cmap="gray")
plt.title("Skeletonization")
plt.axis("off")

plt.tight_layout()
plt.show()
