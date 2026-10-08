import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("img_3.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# Blur levels
blur_levels = [25, 50, 75, 100]

plt.figure(figsize=(12, 12))

for i, level in enumerate(blur_levels):

    # Kernel size must be odd
    kernel = level
    if kernel % 2 == 0:
        kernel += 1

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (kernel, kernel), 0)

    # 2D Fourier Transform
    f = np.fft.fft2(blurred)

    # Shift low frequencies to center
    f_shift = np.fft.fftshift(f)

    # Magnitude spectrum
    magnitude = np.abs(f_shift)

    # Log scale for visualization
    ft = np.log(1 + magnitude)

    # Show blurred image
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(blurred, cmap="gray")
    plt.title(f"Blur Level = {level}")
    plt.axis("off")

    # Show Fourier Transform
    plt.subplot(4, 2, 2 * i + 2)
    plt.imshow(ft, cmap="gray")
    plt.title(f"FT - Blur Level = {level}")
    plt.axis("off")

plt.tight_layout()
plt.show()
