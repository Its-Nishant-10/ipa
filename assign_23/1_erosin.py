import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def erosion(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)

    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i : i + kh, j : j + kw]

            if np.all(region[kernel == 1] == 255):
                output[i, j] = 255

    return output


image = Image.open("img_2.jpeg").convert("L")
image = np.array(image)

binary = np.where(image > 127, 255, 0).astype(np.uint8)

kernel = np.ones((3, 3), dtype=np.uint8)

result = erosion(binary, kernel)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Eroded Image")
plt.axis("off")

plt.tight_layout()
plt.show()

Image.fromarray(result).save("eroded_image.png")
