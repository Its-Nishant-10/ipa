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
# Create motion blur (degradation)
# --------------------------------
def motion_blur_kernel(size=15):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1
    kernel /= size
    return kernel


psf = motion_blur_kernel(15)

# Apply blur
blurred = cv2.filter2D(img, -1, psf)


# --------------------------------
# Inverse Filtering
# --------------------------------
def inverse_filter(blurred, psf, threshold=0.01):

    # Fourier transform of blurred image
    G = np.fft.fft2(blurred)

    # Fourier transform of PSF
    H = np.fft.fft2(psf, s=blurred.shape)

    # Avoid division by very small values
    H[np.abs(H) < threshold] = threshold

    # Inverse filtering
    F = G / H

    # Inverse Fourier transform
    restored = np.fft.ifft2(F)

    # Take real part
    restored = np.real(restored)

    # Normalize
    restored = cv2.normalize(restored, None, 0, 1, cv2.NORM_MINMAX)

    return restored


restored = inverse_filter(blurred, psf)


# --------------------------------
# Display
# --------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap="gray")
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(restored, cmap="gray")
plt.title("Inverse Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()
