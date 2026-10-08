import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_4.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(img.astype(np.float32))

rx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
ry = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)

gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)

J11 = rx**2 + gx**2 + bx**2
J22 = ry**2 + gy**2 + by**2
J12 = rx * ry + gx * gy + bx * by

theta = 0.5 * np.arctan2(2 * J12, J11 - J22)

G = np.sqrt(
    0.5 * (J11 + J22 + (J11 - J22) * np.cos(2 * theta) + 2 * J12 * np.sin(2 * theta))
)

G = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX)
G = G.astype(np.uint8)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(G, cmap="gray")
plt.title("Di Zenzo Edge Detection")
plt.axis("off")

plt.show()
