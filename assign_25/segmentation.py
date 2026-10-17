import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_1.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("img_1.jpeg not found")

# Smooth image to reduce noise
blur = cv2.GaussianBlur(img, (3, 3), 0)

# 1. Point detection using Laplacian
laplacian = cv2.Laplacian(blur, cv2.CV_64F)
point_response = np.uint8(
    cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX)
)
_, points = cv2.threshold(point_response, 40, 255, cv2.THRESH_BINARY)

# 2. Line detection using directional kernels
horizontal_kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)

vertical_kernel = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)

diagonal_kernel_45 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)

diagonal_kernel_135 = np.array(
    [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32
)

responses = [
    np.abs(cv2.filter2D(blur, cv2.CV_32F, kernel))
    for kernel in (
        horizontal_kernel,
        vertical_kernel,
        diagonal_kernel_45,
        diagonal_kernel_135,
    )
]

line_response = np.maximum.reduce(responses)
line_response = cv2.normalize(line_response, None, 0, 255, cv2.NORM_MINMAX).astype(
    np.uint8
)

_, lines = cv2.threshold(line_response, 100, 255, cv2.THRESH_BINARY)

# 3. Edge detection using Canny
edges = cv2.Canny(blur, 50, 150)

# Display results
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(points, cmap="gray")
plt.title("Point Detection")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(lines, cmap="gray")
plt.title("Line Detection")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(edges, cmap="gray")
plt.title("Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.show()
