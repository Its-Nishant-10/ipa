import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img_2.jpeg")

if img is None:
    raise FileNotFoundError("img_1.jpeg not found")

h, w = img.shape[:2]
mid_y, mid_x = h // 2, w // 2

top_left = img[:mid_y, :mid_x]
top_right = img[:mid_y, mid_x:]
bottom_left = img[mid_y:, :mid_x]
bottom_right = img[mid_y:, mid_x:]

titles = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
images = [top_left, top_right, bottom_left, bottom_right]

plt.figure(figsize=(10, 8))

for i, (part, title) in enumerate(zip(images, titles), 1):
    plt.subplot(2, 2, i)
    plt.imshow(cv2.cvtColor(part, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
