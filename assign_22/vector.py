import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans


def vector_quantization(image, block_size=4, codebook_size=16):
    h, w = image.shape

    h = h - (h % block_size)
    w = w - (w % block_size)

    image = image[:h, :w]

    blocks = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i : i + block_size, j : j + block_size]
            blocks.append(block.flatten())

    blocks = np.array(blocks, dtype=np.float32)

    kmeans = KMeans(n_clusters=codebook_size, random_state=42, n_init=10)

    labels = kmeans.fit_predict(blocks)
    codebook = kmeans.cluster_centers_

    reconstructed = np.zeros((h, w), dtype=np.float32)

    k = 0

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = codebook[labels[k]].reshape(block_size, block_size)
            reconstructed[i : i + block_size, j : j + block_size] = block
            k += 1

    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    mse = np.mean((image.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    rmse = np.sqrt(mse)

    original_bits = h * w * 8
    index_bits = int(np.ceil(np.log2(codebook_size)))
    codebook_bits = codebook_size * block_size * block_size * 8
    compressed_bits = len(labels) * index_bits + codebook_bits

    compression_ratio = original_bits / compressed_bits

    return image, reconstructed, codebook, labels, mse, rmse, compression_ratio


image = Image.open("img_2.jpeg").convert("L")
image = np.array(image)

original, reconstructed, codebook, labels, mse, rmse, compression_ratio = (
    vector_quantization(image, block_size=4, codebook_size=16)
)

print("Original Image Size:", original.shape)
print("Block Size: 4 x 4")
print("Codebook Size:", len(codebook))
print("Number of Blocks:", len(labels))
print("MSE:", mse)
print("RMSE:", rmse)
print("Compression Ratio:", compression_ratio)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap="gray")
plt.title("VQ Compressed Image")
plt.axis("off")

plt.tight_layout()
plt.show()

Image.fromarray(reconstructed).save("vq_compressed.png")
