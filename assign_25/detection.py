import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("img_2.jpeg")
if image is None:
    raise FileNotFoundError(
        "Image img_1.jpeg not found in the working directory"
    )
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
point_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)
point_response = cv2.filter2D(
    gray, cv2.CV_32F, point_kernel
)
point_response = np.abs(point_response)
_, point_mask = cv2.threshold(
    point_response, 80, 255, cv2.THRESH_BINARY
)
point_mask = point_mask.astype(np.uint8)
point_result = image.copy()
point_result[point_mask > 0] = [0, 0, 255]  # BGR red
horizontal_kernel = np.array([
    [-1, -1, -1],
    [ 2,  2,  2],
    [-1, -1, -1]
], dtype=np.float32)
vertical_kernel = np.array([
    [-1,  2, -1],
    [-1,  2, -1],
    [-1,  2, -1]
], dtype=np.float32)
diagonal_45_kernel = np.array([
    [-1, -1,  2],
    [-1,  2, -1],
    [ 2, -1, -1]
], dtype=np.float32)
diagonal_135_kernel = np.array([
    [ 2, -1, -1],
    [-1,  2, -1],
    [-1, -1,  2]
], dtype=np.float32)
kernels = [
    ("Horizontal Line Detection", horizontal_kernel),
    ("Vertical Line Detection", vertical_kernel),
    ("Diagonal 45 Line Detection", diagonal_45_kernel),
    ("Diagonal 135 Line Detection", diagonal_135_kernel)
]
line_results = []
for name, kernel in kernels:
    response = cv2.filter2D(
        gray, cv2.CV_32F, kernel
    )
    response = np.abs(response)
    response = cv2.normalize(
        response, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, detected = cv2.threshold(
        response, 100, 255, cv2.THRESH_BINARY
    )
    line_results.append((name, detected))
plt.figure(figsize=(10, 8))
plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.subplot(3, 3, 2)
plt.imshow(point_mask, cmap="gray")
plt.title("Point Detection")
plt.axis("off")
plt.subplot(3, 3, 3)
plt.imshow(cv2.cvtColor(point_result, cv2.COLOR_BGR2RGB))
plt.title("Detected Points in Red")
plt.axis("off")
for i, (name, result) in enumerate(line_results):
    plt.subplot(3, 3, i + 4)
    plt.imshow(result, cmap="gray")
    plt.title(name)
    plt.axis("off")
plt.tight_layout()
plt.show()