import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_4.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

hsi = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

lower = np.array([0, 50, 50])
upper = np.array([30, 255, 255])

mask = cv2.inRange(hsi, lower, upper)

segmented = cv2.bitwise_and(img, img, mask=mask)

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Segmentation Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(segmented)
plt.title("Segmented Image")
plt.axis("off")

plt.show()
