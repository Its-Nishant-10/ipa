from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("img_2.jpg").convert("L")
img = np.array(image, dtype=np.float64)
img = img[:64, :64]
M, N = img.shape
print("Image size:", M, "x", N)
F = np.zeros((M, N), dtype=complex)
for u in range(M):
    for v in range(N):
        total = 0j
        for x in range(M):
            for y in range(N):
                angle = -2j * np.pi * ((u * x / M) + (v * y / N))
                total += img[x, y] * np.exp(angle)
        F[u, v] = total
magnitude = np.abs(F)
print("DFT calculated.")
percentage = 10
number_to_keep = int(M * N * percentage / 100)
magnitudes = magnitude.flatten()
sorted_magnitudes = np.sort(magnitudes)
threshold = sorted_magnitudes[-number_to_keep]
compressed_F = np.zeros_like(F)
for u in range(M):
    for v in range(N):
        if abs(F[u, v]) >= threshold:
            compressed_F[u, v] = F[u, v]
reconstructed = np.zeros((M, N), dtype=complex)
for x in range(M):
    for y in range(N):
        total = 0j
        for u in range(M):
            for v in range(N):
                angle = 2j * np.pi * ((u * x / M) + (v * y / N))
                total += compressed_F[u, v] * np.exp(angle)
        reconstructed[x, y] = total / (M * N)
reconstructed = np.real(reconstructed)
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
Image.fromarray(reconstructed).save("DFT_compressed_reconstructed.png")
print("DFT compression and decompression completed.")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap="gray")
plt.title("DFT Reconstructed")
plt.axis("off")
plt.show()
