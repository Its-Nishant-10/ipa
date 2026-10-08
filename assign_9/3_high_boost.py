import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("img_3.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not read image.")
    exit()
blur = cv2.GaussianBlur(image, (9, 9), 2)
mask = cv2.subtract(image, blur)
A_values = [1, 1.5, 2, 3]
results = []
for A in A_values:
    high_boost = cv2.addWeighted(image, A, mask, 1, 0)
    results.append(high_boost)
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(blur, cmap="gray")
plt.title("Blurred Image")
plt.axis("off")
for i in range(4):
    plt.subplot(2, 3, i + 3)
    plt.imshow(results[i], cmap="gray")
    plt.title(f"High Boost (A={A_values[i]})")
    plt.axis("off")

plt.tight_layout()
plt.show()
