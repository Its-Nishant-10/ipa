import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_4.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gaussian = img.astype(np.float32) + np.random.normal(0, 25, img.shape)
gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)

salt_pepper = img.copy()
prob = 0.05
random = np.random.rand(*img.shape[:2])
salt_pepper[random < prob / 2] = 255
salt_pepper[random > 1 - prob / 2] = 0

exponential = img.astype(np.float32) + np.random.exponential(25, img.shape)
exponential = np.clip(exponential, 0, 255).astype(np.uint8)

gaussian_filtered = cv2.GaussianBlur(gaussian, (5, 5), 0)
salt_pepper_filtered = cv2.medianBlur(salt_pepper, 5)
exponential_filtered = cv2.GaussianBlur(exponential, (5, 5), 0)

plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gaussian)
plt.title("Gaussian Noise")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(salt_pepper)
plt.title("Salt & Pepper Noise")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(exponential)
plt.title("Exponential Noise")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(gaussian_filtered)
plt.title("Gaussian Filtered")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(salt_pepper_filtered)
plt.title("Median Filtered")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(exponential_filtered)
plt.title("Exponential Filtered")
plt.axis("off")

plt.show()
