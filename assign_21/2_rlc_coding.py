from PIL import Image
import numpy as np


def rlc_encode(pixels):
    encoded = []
    current = pixels[0]
    count = 1
    for pixel in pixels[1:]:
        if pixel == current:
            count += 1
        else:
            encoded.append((int(current), count))
            current = pixel
            count = 1
    encoded.append((int(current), count))
    return encoded


def rlc_decode(encoded):
    decoded = []
    for pixel, count in encoded:
        decoded.extend([pixel] * count)
    return np.array(decoded, dtype=np.uint8)


image = Image.open("img_2.jpg").convert("L")
img_array = np.array(image)
pixels = img_array.flatten()
print("Original image shape:", img_array.shape)
print("Original number of pixels:", len(pixels))
encoded = rlc_encode(pixels)
print("Number of RLC pairs:", len(encoded))
print("\nFirst 20 RLC pairs:")
print(encoded[:20])
decoded_pixels = rlc_decode(encoded)
decoded_image = decoded_pixels.reshape(img_array.shape)
Image.fromarray(decoded_image).save("rlc_reconstructed.png")
if np.array_equal(img_array, decoded_image):
    print("\nRLC decoding successful.")
    print("Original and reconstructed images are identical.")
else:
    print("\nError in decoding.")
original_bits = len(pixels) * 8
compressed_bits = len(encoded) * (8 + 32)
compression_ratio = original_bits / compressed_bits
print("\nOriginal bits:", original_bits)
print("Compressed bits:", compressed_bits)
print("Compression ratio:", compression_ratio)
