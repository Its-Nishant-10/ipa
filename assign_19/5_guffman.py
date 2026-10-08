import cv2
import heapq
from collections import Counter
import matplotlib.pyplot as plt

img = cv2.imread("img_4.jpeg", 0)

freq = Counter(img.flatten())

heap = [[f, [s, ""]] for s, f in freq.items()]
heapq.heapify(heap)

while len(heap) > 1:
    lo = heapq.heappop(heap)
    hi = heapq.heappop(heap)

    for pair in lo[1:]:
        pair[1] = "0" + pair[1]

    for pair in hi[1:]:
        pair[1] = "1" + pair[1]

    heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

codes = dict(heap[0][1:])

encoded = "".join(codes[pixel] for pixel in img.flatten())

original_bits = img.size * 8
compressed_bits = len(encoded)

compression_ratio = original_bits / compressed_bits

print("Original Size:", original_bits, "bits")
print("Compressed Size:", compressed_bits, "bits")
print("Compression Ratio:", compression_ratio)

plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()
