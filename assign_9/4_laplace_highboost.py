import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("img_1.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not read image.")
    exit()
A_values = [1, 2, 3, 5]
plt.figure(figsize=(14, 10))
plt.subplot(3, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
for i, A in enumerate(A_values):
    kernel1 = np.array([[0, -1, 0], [-1, A + 4, -1], [0, -1, 0]], dtype=np.float32)
    output = cv2.filter2D(image, -1, kernel1)
    plt.subplot(3, 3, i + 2)
    plt.imshow(output, cmap="gray")
    plt.title(f"4-Nbr A={A}")
    plt.axis("off")
for i, A in enumerate(A_values):
    kernel2 = np.array([[-1, -1, -1], [-1, A + 8, -1], [-1, -1, -1]], dtype=np.float32)
    output = cv2.filter2D(image, -1, kernel2)
    plt.subplot(3, 3, i + 6)
    plt.imshow(output, cmap="gray")
    plt.title(f"8-Nbr A={A}")
    plt.axis("off")
plt.tight_layout()
plt.show()
