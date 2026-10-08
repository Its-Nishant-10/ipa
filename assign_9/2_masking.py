import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("img_3.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not read image.")
    exit()
blur = cv2.GaussianBlur(image, (9, 9), 2)
mask = cv2.subtract(image, blur)
sharpened = cv2.addWeighted(image, 1.0, mask, 1.0, 0)
plt.figure(figsize=(10, 8))
titles = ["Original Image", "Blurred Image", "Unsharp Mask", "Sharpened Image"]
images = [image, blur, mask, sharpened]
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
