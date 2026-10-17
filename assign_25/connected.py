import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_2.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("img_1.jpeg not found")

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

output = np.zeros((*labels.shape, 3), dtype=np.uint8)
np.random.seed(42)

for i in range(1, num_labels):
    output[labels == i] = np.random.randint(50, 256, 3)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title(f"Connected Components: {num_labels - 1}")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Number of connected components:", num_labels - 1)
