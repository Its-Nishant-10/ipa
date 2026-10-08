import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
img = cv2.imread("image_2.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found!")

# Convert image to float
img_float = img.astype(np.float32)
img_norm = img_float / 255.0

# -------------------------
# Identity
# -------------------------
identity = img.copy()

# -------------------------
# Negative
# -------------------------
negative = 255 - img

# -------------------------
# Log Transformation
# -------------------------
c = 255 / np.log(1 + np.max(img_float))

log_img = c * np.log(1 + img_float)
log_img = np.clip(log_img, 0, 255).astype(np.uint8)

# -------------------------
# Inverse Log
# -------------------------
inv_log = np.exp(img_norm) - 1
inv_log = inv_log / np.max(inv_log)
inv_log = np.clip(inv_log * 255, 0, 255).astype(np.uint8)

# -------------------------
# Nth Root Transformation
# -------------------------
n = 2
root_img = np.power(img_norm, 1 / n)
root_img = np.clip(root_img * 255, 0, 255).astype(np.uint8)

# -------------------------
# Gamma Transformation
# -------------------------
gamma = 2.0

gamma_img = np.power(img_norm, gamma)
gamma_img = np.clip(gamma_img * 255, 0, 255).astype(np.uint8)

# -------------------------
# Display
# -------------------------
images = [img, identity, negative, log_img, inv_log, root_img, gamma_img]

titles = [
    "Original",
    "Identity",
    "Negative",
    "Log",
    "Inverse Log",
    "Nth Root",
    "Gamma (γ = 2.0)",
]

plt.figure(figsize=(14, 8))

for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
