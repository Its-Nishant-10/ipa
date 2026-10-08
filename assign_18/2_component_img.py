import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_4.jpeg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]
rgb_float = rgb.astype(float) / 255.0
R_norm = rgb_float[:, :, 0]
G_norm = rgb_float[:, :, 1]
B_norm = rgb_float[:, :, 2]
K = 1 - np.maximum.reduce([R_norm, G_norm, B_norm])
denominator = 1 - K
denominator[denominator == 0] = 1
C = (1 - R_norm - K) / denominator
M = (1 - G_norm - K) / denominator
Y = (1 - B_norm - K) / denominator
C = (C * 255).astype(np.uint8)
M = (M * 255).astype(np.uint8)
Y = (Y * 255).astype(np.uint8)
K_display = (K * 255).astype(np.uint8)
R = R_norm
G = G_norm
B = B_norm
I = (R + G + B) / 3
minimum = np.minimum.reduce([R, G, B])
S = 1 - (3 / (R + G + B + 1e-10)) * minimum
numerator = 0.5 * ((R - G) + (R - B))
denominator_h = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
theta = np.arccos(np.clip(numerator / (denominator_h + 1e-10), -1, 1))
H = np.where(B <= G, theta, 2 * np.pi - theta)
H = H * 180 / np.pi
H_display = (H / 360 * 255).astype(np.uint8)
S_display = (S * 255).astype(np.uint8)
I_display = (I * 255).astype(np.uint8)
plt.figure(figsize=(12, 10))
plt.subplot(3, 3, 1)
plt.imshow(R * 255, cmap="gray")
plt.title("R Component")
plt.axis("off")
plt.subplot(3, 3, 2)
plt.imshow(G * 255, cmap="gray")
plt.title("G Component")
plt.axis("off")
plt.subplot(3, 3, 3)
plt.imshow(B * 255, cmap="gray")
plt.title("B Component")
plt.axis("off")
plt.subplot(3, 3, 4)
plt.imshow(C, cmap="gray")
plt.title("C Component")
plt.axis("off")
plt.subplot(3, 3, 5)
plt.imshow(M, cmap="gray")
plt.title("M Component")
plt.axis("off")
plt.subplot(3, 3, 6)
plt.imshow(Y, cmap="gray")
plt.title("Y Component")
plt.axis("off")
plt.subplot(3, 3, 7)
plt.imshow(H_display, cmap="gray")
plt.title("H Component")
plt.axis("off")
plt.subplot(3, 3, 8)
plt.imshow(S_display, cmap="gray")
plt.title("S Component")
plt.axis("off")
plt.subplot(3, 3, 9)
plt.imshow(I_display, cmap="gray")
plt.title("I Component")
plt.axis("off")
plt.tight_layout()
plt.show()
