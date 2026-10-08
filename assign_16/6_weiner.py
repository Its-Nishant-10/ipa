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
# Create Motion Blur (PSF)
# --------------------------------
def motion_blur_kernel(size=15):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1
    kernel /= size
    return kernel


psf = motion_blur_kernel(15)


# --------------------------------
# Degrade Image
# --------------------------------
blurred = cv2.filter2D(img, -1, psf)

# Add Gaussian noise
noise = np.random.normal(0, 0.03, blurred.shape)
degraded = blurred + noise
degraded = np.clip(degraded, 0, 1)


# --------------------------------
# Wiener Filter
# --------------------------------
def wiener_filter(degraded, psf, K=0.01):

    # Fourier transform of degraded image
    G = np.fft.fft2(degraded)

    # Fourier transform of PSF
    H = np.fft.fft2(psf, s=degraded.shape)

    # Wiener filter
    H_conj = np.conj(H)

    W = H_conj / (np.abs(H) ** 2 + K)

    # Apply Wiener filter
    F_hat = W * G

    # Inverse Fourier transform
    restored = np.fft.ifft2(F_hat)

    restored = np.real(restored)

    # Normalize
    restored = cv2.normalize(restored, None, 0, 1, cv2.NORM_MINMAX)

    return restored


# Apply Wiener filter
restored = wiener_filter(degraded, psf, K=0.01)


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
plt.title("Wiener Filtered Image")
plt.axis("off")

plt.tight_layout()
plt.show()
