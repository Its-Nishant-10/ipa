import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("img_1.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found")


# -----------------------------
# Geometric Mean Filter
# -----------------------------
def geometric_mean_filter(image, size=3):
    image = image.astype(np.float32) + 1e-5

    log_image = np.log(image)

    kernel = np.ones((size, size), np.float32) / (size * size)

    filtered = np.exp(cv2.filter2D(log_image, -1, kernel))

    return np.clip(filtered, 0, 255).astype(np.uint8)


# -----------------------------
# Harmonic Mean Filter
# -----------------------------
def harmonic_mean_filter(image, size=3):
    image = image.astype(np.float32) + 1e-5

    kernel = np.ones((size, size), np.float32)

    numerator = size * size

    denominator = cv2.filter2D(1.0 / image, -1, kernel)

    filtered = numerator / denominator

    return np.clip(filtered, 0, 255).astype(np.uint8)


# -----------------------------
# Contraharmonic Mean Filter
# -----------------------------
def contraharmonic_mean_filter(image, size=3, Q=1.5):
    image = image.astype(np.float32)

    kernel = np.ones((size, size), np.float32)

    numerator = cv2.filter2D(np.power(image, Q + 1), -1, kernel)

    denominator = cv2.filter2D(np.power(image, Q), -1, kernel)

    denominator = denominator + 1e-5

    filtered = numerator / denominator

    return np.clip(filtered, 0, 255).astype(np.uint8)


# -----------------------------
# Apply filters
# -----------------------------

geometric = geometric_mean_filter(img, 3)

harmonic = harmonic_mean_filter(img, 3)

contraharmonic = contraharmonic_mean_filter(img, size=3, Q=1.5)


# -----------------------------
# Display
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original / Noisy Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(geometric, cmap="gray")
plt.title("Geometric Mean Filter")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(harmonic, cmap="gray")
plt.title("Harmonic Mean Filter")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(contraharmonic, cmap="gray")
plt.title("Contraharmonic Mean Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
