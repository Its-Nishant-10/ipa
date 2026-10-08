import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Read image
# --------------------------------
img = cv2.imread("img_1.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found")

img = img.astype(np.float32) / 255.0


# --------------------------------
# Create motion blur
# --------------------------------
def motion_blur_kernel(size=15):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1
    kernel /= size
    return kernel


psf = motion_blur_kernel(15)

# Blur the image
blurred = cv2.filter2D(img, -1, psf)


# --------------------------------
# Add Gaussian Noise
# --------------------------------
noise = np.random.normal(0, 0.05, blurred.shape)

degraded = blurred + noise
degraded = np.clip(degraded, 0, 1)


# --------------------------------
# Inverse Filtering
# --------------------------------
G = np.fft.fft2(degraded)

H = np.fft.fft2(psf, s=degraded.shape)

# Direct inverse filtering
# Small H values cause huge noise amplification
H[np.abs(H) < 1e-10] = 1e-10

F = G / H

restored = np.fft.ifft2(F)
restored = np.real(restored)

# Normalize for display
restored = cv2.normalize(restored, None, 0, 1, cv2.NORM_MINMAX)


# --------------------------------
# Display Results
# --------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(blurred, cmap="gray")
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(degraded, cmap="gray")
plt.title("Blur + Gaussian Noise")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(restored, cmap="gray")
plt.title("Inverse Filtering Failure")
plt.axis("off")

plt.tight_layout()
plt.show()
