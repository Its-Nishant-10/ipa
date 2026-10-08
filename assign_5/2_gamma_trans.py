import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("image_2.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# Normalize image
img_norm = img.astype(np.float32) / 255.0

# Different gamma values
gamma_values = [0.1, 0.6, 1.5, 2.5, 6.0]

plt.figure(figsize=(15, 8))

# Original image
plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

# Gamma transformed images
for i, gamma in enumerate(gamma_values):
    gamma_img = np.power(img_norm, gamma)
    gamma_img = np.clip(gamma_img * 255, 0, 255).astype(np.uint8)

    plt.subplot(2, 3, i + 2)
    plt.imshow(gamma_img, cmap="gray")
    plt.title(f"Gamma = {gamma}")
    plt.axis("off")

plt.tight_layout()
plt.show()
