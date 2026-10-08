import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image_2.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

lower = 100
upper = 180

slice_img = img.copy()

slice_img[(img >= lower) & (img <= upper)] = 255

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(slice_img, cmap="gray")
plt.title("Contrast Slicing\n(With Background)")
plt.axis("off")

plt.tight_layout()
plt.show()
