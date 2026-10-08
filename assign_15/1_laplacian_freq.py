import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_2.jpeg", cv2.IMREAD_GRAYSCALE)
F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)
M, N = img.shape
u = np.arange(-N // 2, N // 2)
v = np.arange(-M // 2, M // 2)
U, V = np.meshgrid(u, v)
H = -4 * np.pi**2 * (U**2 + V**2)
G = H * F_shift
G_shift = np.fft.ifftshift(G)
laplacian = np.fft.ifft2(G_shift)
laplacian = np.abs(laplacian)
laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian - Frequency Domain")
plt.axis("off")
plt.show()
