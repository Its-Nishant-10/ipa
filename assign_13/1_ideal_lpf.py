import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("img_3.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# 2D Fourier Transform
F = np.fft.fft2(img)

# Shift zero frequency to center
F_shift = np.fft.fftshift(F)

# Image dimensions
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Different filter radii
radii = [10, 25, 50, 100]

plt.figure(figsize=(12, 12))

for i, radius in enumerate(radii):

    # Create ideal low-pass filter
    mask = np.zeros((rows, cols), np.uint8)

    # Draw filled circle at the center
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    # Apply filter
    F_filtered = F_shift * mask

    # Inverse Fourier Transform
    F_ishift = np.fft.ifftshift(F_filtered)
    img_filtered = np.fft.ifft2(F_ishift)

    # Convert to real image
    img_filtered = np.abs(img_filtered)

    # Fourier magnitude spectrum
    spectrum = np.log(1 + np.abs(F_filtered))

    # Filtered image
    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(img_filtered, cmap="gray")
    plt.title(f"ILPF - Radius {radius}")
    plt.axis("off")

    # Filter mask
    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Filter Mask - R={radius}")
    plt.axis("off")

    # Fourier spectrum
    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(spectrum, cmap="gray")
    plt.title(f"FT - Radius {radius}")
    plt.axis("off")

plt.tight_layout()
plt.show()
