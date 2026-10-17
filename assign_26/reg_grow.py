import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

img = cv2.imread("img_2.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("img_1.jpeg not found")


def region_growing(img, seed, threshold=15):
    h, w = img.shape
    visited = np.zeros((h, w), dtype=bool)
    region = np.zeros((h, w), dtype=np.uint8)

    sy, sx = seed
    seed_value = int(img[sy, sx])
    queue = deque([(sy, sx)])
    visited[sy, sx] = True

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        y, x = queue.popleft()
        region[y, x] = 255

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True

                if abs(int(img[ny, nx]) - seed_value) <= threshold:
                    queue.append((ny, nx))

    return region


# Seed pixel: center of image
seed = (img.shape[0] // 2, img.shape[1] // 2)
result = region_growing(img, seed, threshold=15)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.scatter(seed[1], seed[0], color="red", s=30)
plt.title("Original Image and Seed")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Region Growing")
plt.axis("off")

plt.tight_layout()
plt.show()
