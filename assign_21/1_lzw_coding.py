from PIL import Image
import numpy as np


def lzw_encode(data):
    dictionary = {bytes([i]): i for i in range(256)}
    next_code = 256
    w = bytes([data[0]])
    compressed = []
    for pixel in data[1:]:
        c = bytes([pixel])
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            compressed.append(dictionary[w])
            dictionary[wc] = next_code
            next_code += 1
            w = c
    if w:
        compressed.append(dictionary[w])
    return compressed


def lzw_decode(compressed):
    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256
    w = dictionary[compressed[0]]
    output = bytearray(w)
    for code in compressed[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = w + w[:1]
        else:
            raise ValueError("Invalid LZW code")
        output.extend(entry)
        dictionary[next_code] = w + entry[:1]
        next_code += 1
        w = entry
    return np.frombuffer(output, dtype=np.uint8)


image = Image.open("img_2.jpg").convert("L")
img_array = np.array(image)
pixels = img_array.flatten()
print("Original Image Size:", img_array.shape)
print("Number of Pixels:", len(pixels))
compressed = lzw_encode(pixels)
print("Number of LZW Codes:", len(compressed))
decoded_pixels = lzw_decode(compressed)
decoded_image = decoded_pixels.reshape(img_array.shape)
Image.fromarray(decoded_image).save("reconstructed.png")
print("LZW Encoding and Decoding Completed.")
print("Reconstructed image saved as reconstructed.png")
original_bits = len(pixels) * 8
max_code = max(compressed)
bits_per_code = max(1, max_code.bit_length())
compressed_bits = len(compressed) * bits_per_code
compression_ratio = original_bits / compressed_bits
print("\nOriginal bits:", original_bits)
print("Compressed bits:", compressed_bits)
print("Compression Ratio:", compression_ratio)
