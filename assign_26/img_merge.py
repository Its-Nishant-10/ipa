import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img_2.jpeg")

if img is None:
    raise FileNotFoundError("img_1.jpeg not found")

h, w = img.shape[:2]
mid_y, mid_x = h // 2, w // 2

top_left = img[:mid_y, :mid_x]
top_right = img[:mid_y, mid_x:]
bottom_left = img[mid_y:, :mid_x]
bottom_right = img[mid_y:, mid_x:]

# Merge regions horizontally, then vertically
top = np.hstack((top_left, top_right))
bottom = np.hstack((bottom_left, bottom_right))
merged = np.vstack((top, bottom))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
plt.title("Merged Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Images merged successfully:", np.array_equal(img, merged))
