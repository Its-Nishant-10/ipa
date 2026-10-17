import cv2
import matplotlib.pyplot as plt

# 1. Read your own image
img = cv2.imread("img_1.jpeg")

if img is None:
    raise FileNotFoundError("Image not found! Check the filename.")

# 2. Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Convert grayscale to binary
_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

# 4. Create structuring element
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (5, 5)
)

# 5. Apply opening (erosion followed by dilation)
opening = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel
)

# 6. Apply closing (dilation followed by erosion)
closing = cv2.morphologyEx(
    binary, cv2.MORPH_CLOSE, kernel
)

# 7. Display all results
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(opening, cmap="gray")
plt.title("Opening")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(closing, cmap="gray")
plt.title("Closing")
plt.axis("off")

plt.tight_layout()
plt.show()

# 8. Save output images
cv2.imwrite("opening_result.jpg", opening)
cv2.imwrite("closing_result.jpg", closing)