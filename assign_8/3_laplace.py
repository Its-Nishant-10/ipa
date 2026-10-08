import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread("image_1.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not read image.")
    exit()

# ---------------- Laplacian Kernels ----------------

kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

kernel2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

kernel3 = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]], dtype=np.float32)

# ---------------- Apply Filters ----------------

lap1 = cv2.filter2D(image, cv2.CV_32F, kernel1)
lap2 = cv2.filter2D(image, cv2.CV_32F, kernel2)
lap3 = cv2.filter2D(image, cv2.CV_32F, kernel3)

# Convert to displayable format
lap1 = cv2.convertScaleAbs(lap1)
lap2 = cv2.convertScaleAbs(lap2)
lap3 = cv2.convertScaleAbs(lap3)

# ---------------- Display ----------------

plt.figure(figsize=(12, 8))

titles = ["Original", "Laplacian Kernel 1", "Laplacian Kernel 2", "Laplacian Kernel 3"]

images = [image, lap1, lap2, lap3]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
