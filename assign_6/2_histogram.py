import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", 0)
equalized = cv2.equalizeHist(img)
cv2.imwrite("equalized.jpg", equalized)
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.subplot(122)
plt.imshow(equalized, cmap="gray")
plt.title("Equalized")
plt.show()
