import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image_1.jpg", 0)
equalized = cv2.equalizeHist(img)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(2, 2, 2)
plt.hist(img.ravel(), 256, [0, 256])
plt.title("Original Histogram")
plt.subplot(2, 2, 3)
plt.imshow(equalized, cmap="gray")
plt.title("Equalized")
plt.subplot(2, 2, 4)
plt.hist(equalized.ravel(), 256, [0, 256])
plt.title("Equalized Histogram")
plt.show()
