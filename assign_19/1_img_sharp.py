import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_4.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

channels = cv2.split(img)
result = []

for channel in channels:
    laplacian = cv2.filter2D(channel, cv2.CV_32F, kernel)
    sharpened = channel.astype(np.float32) - laplacian
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    result.append(sharpened)

sharpened = cv2.merge(result)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sharpened)
plt.title("Sharpened Image")
plt.axis("off")

plt.show()
