import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Read SINGLE image
# ==========================================
img = cv2.imread("img_3.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# ==========================================
# Fourier Transform
# ==========================================
F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)

# Image dimensions
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# ==========================================
# Distance matrix
# ==========================================
u = np.arange(cols)
v = np.arange(rows)

U, V = np.meshgrid(u, v)

D = np.sqrt((U - ccol) ** 2 + (V - crow) ** 2)

# ==========================================
# Cutoff frequency
# ==========================================
D0 = 50

# Butterworth order
n = 2

# Avoid division by zero
D[D == 0] = 1e-5

# ==========================================
# 1. IDEAL HIGH-PASS FILTER
# ==========================================
H_ideal = np.zeros((rows, cols))

H_ideal[D > D0] = 1

# Apply filter
F_ideal = F_shift * H_ideal

# Inverse FFT
img_ideal = np.fft.ifft2(np.fft.ifftshift(F_ideal))

img_ideal = np.abs(img_ideal)

# ==========================================
# 2. BUTTERWORTH HIGH-PASS FILTER
# ==========================================
H_butterworth = 1 / (1 + (D0 / D) ** (2 * n))

# Apply filter
F_butterworth = F_shift * H_butterworth

# Inverse FFT
img_butterworth = np.fft.ifft2(np.fft.ifftshift(F_butterworth))

img_butterworth = np.abs(img_butterworth)

# ==========================================
# 3. GAUSSIAN HIGH-PASS FILTER
# ==========================================
H_gaussian = 1 - np.exp(-(D**2) / (2 * D0**2))

# Apply filter
F_gaussian = F_shift * H_gaussian

# Inverse FFT
img_gaussian = np.fft.ifft2(np.fft.ifftshift(F_gaussian))

img_gaussian = np.abs(img_gaussian)

# ==========================================
# Fourier Spectrums
# ==========================================
spectrum_ideal = np.log(1 + np.abs(F_ideal))

spectrum_butterworth = np.log(1 + np.abs(F_butterworth))

spectrum_gaussian = np.log(1 + np.abs(F_gaussian))

# ==========================================
# DISPLAY RESULTS
# ==========================================

plt.figure(figsize=(15, 12))

# Original image
plt.subplot(4, 4, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Original Fourier spectrum
plt.subplot(4, 4, 2)
plt.imshow(np.log(1 + np.abs(F_shift)), cmap="gray")
plt.title("Original FT")
plt.axis("off")

# Ideal filter
plt.subplot(4, 4, 3)
plt.imshow(H_ideal, cmap="gray")
plt.title("Ideal HPF")
plt.axis("off")

# Ideal result
plt.subplot(4, 4, 4)
plt.imshow(img_ideal, cmap="gray")
plt.title("Ideal HPF Result")
plt.axis("off")


# Butterworth filter
plt.subplot(4, 4, 5)
plt.imshow(H_butterworth, cmap="gray")
plt.title(f"Butterworth HPF (n={n})")
plt.axis("off")

# Butterworth result
plt.subplot(4, 4, 6)
plt.imshow(img_butterworth, cmap="gray")
plt.title("Butterworth Result")
plt.axis("off")

# Butterworth spectrum
plt.subplot(4, 4, 7)
plt.imshow(spectrum_butterworth, cmap="gray")
plt.title("Butterworth FT")
plt.axis("off")


# Gaussian filter
plt.subplot(4, 4, 9)
plt.imshow(H_gaussian, cmap="gray")
plt.title("Gaussian HPF")
plt.axis("off")

# Gaussian result
plt.subplot(4, 4, 10)
plt.imshow(img_gaussian, cmap="gray")
plt.title("Gaussian Result")
plt.axis("off")

# Gaussian spectrum
plt.subplot(4, 4, 11)
plt.imshow(spectrum_gaussian, cmap="gray")
plt.title("Gaussian FT")
plt.axis("off")

# Ideal spectrum
plt.subplot(4, 4, 12)
plt.imshow(spectrum_ideal, cmap="gray")
plt.title("Ideal HPF FT")
plt.axis("off")

plt.tight_layout()
plt.show()
