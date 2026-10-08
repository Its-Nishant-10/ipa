import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("img_3.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not read image.")
    exit()
kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
kernel2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
output1 = cv2.filter2D(image, -1, kernel1)
output2 = cv2.filter2D(image, -1, kernel2)
plt.figure(figsize=(10, 8))
titles = [
    "Original Image",
    "Kernel 1",
    "Kernel 2",
]
images = [image, output1, output2]
for i in range(3):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
