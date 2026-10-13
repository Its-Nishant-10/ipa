import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read your own image
image = cv2.imread("img_2.jpeg")

if image is None:
    raise FileNotFoundError("Image not found! Check the filename.")

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert grayscale image to binary
_, binary_image = cv2.threshold(
    gray_image, 127, 255, cv2.THRESH_BINARY
)

# Create a structuring element (kernel)
kernel = np.ones((7, 7), dtype=np.uint8)

# Apply dilation
dilated_image = cv2.dilate(
    binary_image, kernel, iterations=1
)

# Display original and dilated images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap="gray")
plt.title("Original Binary Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dilated_image, cmap="gray")
plt.title("Image After Dilation")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the result
cv2.imwrite("dilated_image.jpg", dilated_image)