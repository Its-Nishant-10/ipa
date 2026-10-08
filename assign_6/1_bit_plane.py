import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", 0)
plt.figure(figsize=(12, 6))
for i in range(8):
    plane = (img >> i) & 1
    plt.subplot(2, 4, i + 1)
    plt.imshow(plane, cmap="gray")
    plt.title(f"Bit Plane {i}")
    plt.axis("off")
plt.show()
