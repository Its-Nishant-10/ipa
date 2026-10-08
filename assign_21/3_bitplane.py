from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
image = Image.open("img_2.jpg").convert("L")
img = np.array(image, dtype=np.uint8)
print("Image shape:", img.shape)
print("Minimum pixel value:", img.min())
print("Maximum pixel value:", img.max())
planes = []
for bit in range(8):
    plane = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            shifted = img[i, j] >> bit
            extracted_bit = shifted & 1
            plane[i, j] = extracted_bit
    planes.append(plane)
plt.figure(figsize=(12, 8))
for bit in range(8):
    plt.subplot(2, 4, bit + 1)
    display_plane = planes[bit] * 255
    plt.imshow(display_plane, cmap="gray")
    plt.title("Bit Plane " + str(bit))
    plt.axis("off")
plt.tight_layout()
plt.show()
reconstructed = np.zeros(img.shape, dtype=np.uint8)
for bit in range(8):
    reconstructed += planes[bit] * (2**bit)
if np.array_equal(img, reconstructed):
    print("Image reconstructed successfully.")
    print("Original and reconstructed images are identical.")
else:
    print("Reconstruction error.")
reconstructed_image = Image.fromarray(reconstructed)
reconstructed_image.save("reconstructed_image.png")
lossy_image = (
    planes[7] * 128 + planes[6] * 64 + planes[5] * 32 + planes[4] * 16
).astype(np.uint8)
Image.fromarray(lossy_image).save("bit_plane_compressed.png")
print("Lossy bit-plane image saved.")
