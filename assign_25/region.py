import cv2
import numpy as np
import matplotlib.pyplot as plt
image = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(image, (10, 100), 150, 255, 3)
filled = image.copy()
mask = np.zeros((202, 202), dtype=np.uint8)
cv2.floodFill(filled, mask, (10, 100), 255)
result = cv2.bitwise_or(image, filled)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(filled, cmap="gray")
plt.title("After Flood Fill")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result, cmap="gray")
plt.title("Region Filled")
plt.axis("off")

plt.tight_layout()
plt.show()