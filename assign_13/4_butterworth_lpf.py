import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Read single image
# =========================
img = cv2.imread("img_3.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# =========================
# Fourier Transform
# =========================
F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)

# Image dimensions
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# =========================
# Distance matrix D(u,v)
# =========================
u = np.arange(cols)
v = np.arange(rows)

U, V = np.meshgrid(u, v)

D = np.sqrt((U - ccol) ** 2 + (V - crow) ** 2)

# =========================
# Cutoff frequency
# =========================
D0 = 50

# Different Butterworth orders
orders = [1, 2, 4, 8]

plt.figure(figsize=(12, 10))

for i, n in enumerate(orders):

    # =========================
    # Butterworth Low Pass Filter
    # =========================
    H = 1 / (1 + (D / D0) ** (2 * n))

    # Apply filter
    F_filtered = F_shift * H

    # =========================
    # Inverse Fourier Transform
    # =========================
    F_ishift = np.fft.ifftshift(F_filtered)

    img_filtered = np.fft.ifft2(F_ishift)

    img_filtered = np.abs(img_filtered)

    # =========================
    # Fourier spectrum
    # =========================
    spectrum = np.log(1 + np.abs(F_filtered))

    # =========================
    # Display filtered image
    # =========================
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(img_filtered, cmap="gray")
    plt.title(f"Butterworth LPF, n = {n}")
    plt.axis("off")

    # =========================
    # Display filter
    # =========================
    plt.subplot(4, 2, 2 * i + 2)
    plt.imshow(H, cmap="gray")
    plt.title(f"Filter, n = {n}")
    plt.axis("off")

plt.tight_layout()
plt.show()
