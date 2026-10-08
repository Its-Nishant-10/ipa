import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image_2.jpg", 0)
average_kernel = np.ones((3, 3), np.float32) / 9
weighted_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
average = cv2.filter2D(img, -1, average_kernel)
weighted = cv2.filter2D(img, -1, weighted_kernel)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(132)
plt.imshow(average, cmap="gray")
plt.title("Average Filter")
plt.subplot(133)
plt.imshow(weighted, cmap="gray")
plt.title("Weighted Filter")
plt.show()
