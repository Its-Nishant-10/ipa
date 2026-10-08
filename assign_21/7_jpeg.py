import heapq
from collections import Counter
import numpy as np
from PIL import Image

Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

n = 8
x = np.arange(n)
s = np.full(n, np.sqrt(2 / n))
s[0] = np.sqrt(1 / n)

C = s[:, None] * np.cos(np.pi * (2 * x + 1) * x[:, None] / (2 * n))


def zigzag(a):
    return [
        a[i, s - i]
        for s in range(15)
        for i in (
            range(min(s, 7), max(-1, s - 8), -1)
            if s % 2 == 0
            else range(max(0, s - 7), min(s, 7) + 1)
        )
    ]


def unzigzag(values):
    a = np.zeros((8, 8), dtype=int)
    k = 0

    for s in range(15):
        rows = (
            range(min(s, 7), max(-1, s - 8), -1)
            if s % 2 == 0
            else range(max(0, s - 7), min(s, 7) + 1)
        )

        for i in rows:
            a[i, s - i] = values[k]
            k += 1

    return a


def rle(values):
    result = []
    start = 0

    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] != values[start]:
            result.append((int(values[start]), end - start))
            start = end

    return result


def huffman(values):
    counts = Counter(values)

    if len(counts) == 1:
        value = next(iter(counts))
        return "0" * counts[value], {value: "0"}

    heap = [[count, [value, ""]] for value, count in counts.items()]

    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        for item in left[1:]:
            item[1] = "0" + item[1]

        for item in right[1:]:
            item[1] = "1" + item[1]

        heapq.heappush(heap, [left[0] + right[0], *left[1:], *right[1:]])

    codes = dict(heap[0][1:])

    bits = "".join(codes[value] for value in values)

    return bits, codes


def block_encode(block):
    dct = np.round(C @ (block - 128) @ C.T / Q).astype(int)

    return rle(zigzag(dct))


def block_decode(data):
    values = np.repeat(
        [value for value, count in data], [count for value, count in data]
    )

    values = values[:64]

    if len(values) < 64:
        values = np.pad(values, (0, 64 - len(values)))

    return np.clip(C.T @ (unzigzag(values) * Q) @ C + 128, 0, 255)


def compress(image):
    image = np.asarray(image, dtype=float)

    h, w = image.shape

    ph = (h + 7) // 8 * 8
    pw = (w + 7) // 8 * 8

    padded = np.zeros((ph, pw))
    padded[:h, :w] = image

    blocks = []

    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            block = padded[i : i + 8, j : j + 8]
            blocks.append(block_encode(block))

    data = [pair for block in blocks for pair in block]

    bits, codes = huffman(data)

    return bits, codes, (ph, pw), blocks


def decompress(blocks, shape, padded_shape):
    output = np.zeros(padded_shape)

    k = 0

    for i in range(0, padded_shape[0], 8):
        for j in range(0, padded_shape[1], 8):
            output[i : i + 8, j : j + 8] = block_decode(blocks[k])
            k += 1

    return output[: shape[0], : shape[1]]


image = Image.open("img_2.jpg").convert("L")

original = np.asarray(image)

bits, codes, padded_shape, blocks = compress(original)

reconstructed = decompress(blocks, original.shape, padded_shape)

Image.fromarray(reconstructed.astype(np.uint8)).save("jpeg_decompressed.png")

rms = np.sqrt(np.mean((original.astype(float) - reconstructed) ** 2))

print("Original:", original.shape)
print("Compressed bits:", len(bits))
print("RMS error:", rms)
