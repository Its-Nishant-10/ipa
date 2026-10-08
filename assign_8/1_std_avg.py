import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("image_1.jpg")

if image is None:
    print("Error: Could not read image.")
    exit()

# Convert BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Average Blur
blur_5x5 = cv2.blur(image_rgb, (5, 5))
blur_7x7 = cv2.blur(image_rgb, (7, 7))
blur_15x15 = cv2.blur(image_rgb, (15, 15))

# Display all images together
plt.figure(figsize=(12, 8))

titles = [
    "Original Image",
    "Average Blur (5x5)",
    "Average Blur (7x7)",
    "Average Blur (15x15)",
]

images = [image_rgb, blur_5x5, blur_7x7, blur_15x15]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
