from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
image = Image.open("img_2.jpg").convert("L")
img = np.array(image, dtype=np.float64)
img = img[:64, :64]
M, N = img.shape
print("Image size:", M, "x", N)
DCT = np.zeros((M, N))
for u in range(M):
    for v in range(N):
        if u == 0:
            alpha_u = np.sqrt(1 / M)
        else:
            alpha_u = np.sqrt(2 / M)
        if v == 0:
            alpha_v = np.sqrt(1 / N)
        else:
            alpha_v = np.sqrt(2 / N)
        total = 0
        for x in range(M):
            for y in range(N):
                cos_x = np.cos(np.pi * (2 * x + 1) * u / (2 * M))
                cos_y = np.cos(np.pi * (2 * y + 1) * v / (2 * N))
                total += img[x, y] * cos_x * cos_y
        DCT[u, v] = alpha_u * alpha_v * total
print("DCT calculated.")
keep = 16
compressed_DCT = np.zeros_like(DCT)
compressed_DCT[:keep, :keep] = DCT[:keep, :keep]
reconstructed = np.zeros((M, N))
for x in range(M):
    for y in range(N):
        total = 0
        for u in range(M):
            for v in range(N):
                if u == 0:
                    alpha_u = np.sqrt(1 / M)
                else:
                    alpha_u = np.sqrt(2 / M)
                if v == 0:
                    alpha_v = np.sqrt(1 / N)
                else:
                    alpha_v = np.sqrt(2 / N)
                cos_x = np.cos(np.pi * (2 * x + 1) * u / (2 * M))
                cos_y = np.cos(np.pi * (2 * y + 1) * v / (2 * N))
                total += alpha_u * alpha_v * compressed_DCT[u, v] * cos_x * cos_y
        reconstructed[x, y] = total
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
Image.fromarray(reconstructed).save("DCT_compressed_reconstructed.png")
print("DCT compression and decompression completed.")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap="gray")
plt.title("DCT Reconstructed")
plt.axis("off")
plt.show()