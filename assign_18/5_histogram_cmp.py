import cv2
import numpy as np
import matplotlib.pyplot as plt


def rgb_to_hsi(img):
    rgb = img.astype(np.float32) / 255.0
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    I = (R + G + B) / 3.0
    minimum = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 * minimum / (R + G + B + 1e-10))
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(np.clip(numerator / (denominator + 1e-10), -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H * 180 / np.pi
    return H, S, I


def hsi_to_rgb(H, S, I):
    H = H % 360
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    sector1 = H < 120
    B[sector1] = I[sector1] * (1 - S[sector1])
    R[sector1] = I[sector1] * (
        1
        + (
            S[sector1]
            * np.cos(np.radians(H[sector1]))
            / np.cos(np.radians(60 - H[sector1]))
        )
    )
    G[sector1] = 3 * I[sector1] - (R[sector1] + B[sector1])
    sector2 = (H >= 120) & (H < 240)
    H2 = H[sector2] - 120
    R[sector2] = I[sector2] * (1 - S[sector2])
    G[sector2] = I[sector2] * (
        1 + (S[sector2] * np.cos(np.radians(H2)) / np.cos(np.radians(60 - H2)))
    )
    B[sector2] = 3 * I[sector2] - (R[sector2] + G[sector2])
    sector3 = H >= 240
    H3 = H[sector3] - 240
    G[sector3] = I[sector3] * (1 - S[sector3])
    B[sector3] = I[sector3] * (
        1 + (S[sector3] * np.cos(np.radians(H3)) / np.cos(np.radians(60 - H3)))
    )
    R[sector3] = 3 * I[sector3] - (G[sector3] + B[sector3])
    rgb = np.stack((R, G, B), axis=2)
    rgb = np.clip(rgb * 255, 0, 255)
    return rgb.astype(np.uint8)


img_bgr = cv2.imread("img_6.jpeg")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]
R_eq = cv2.equalizeHist(R)
G_eq = cv2.equalizeHist(G)
B_eq = cv2.equalizeHist(B)
rgb_equalized = cv2.merge([R_eq, G_eq, B_eq])
H, S, I = rgb_to_hsi(img)
I_8bit = np.clip(I * 255, 0, 255).astype(np.uint8)
I_eq_8bit = cv2.equalizeHist(I_8bit)
I_eq = I_eq_8bit.astype(np.float32) / 255.0
hsi_equalized = hsi_to_rgb(H, S, I_eq)
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(rgb_equalized)
plt.title("RGB Histogram Equalization")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(hsi_equalized)
plt.title("HSI Histogram Equalization")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.hist(I.ravel(), 256, range=(0, 1))
plt.title("Original Intensity Histogram")
plt.subplot(2, 3, 5)
plt.hist(I_eq.ravel(), 256, range=(0, 1))
plt.title("HSI Equalized Intensity")
plt.subplot(2, 3, 6)
plt.hist(R_eq.ravel(), 256, range=(0, 255), alpha=0.4)
plt.hist(G_eq.ravel(), 256, range=(0, 255), alpha=0.4)
plt.hist(B_eq.ravel(), 256, range=(0, 255), alpha=0.4)
plt.title("Equalized RGB Channels")
plt.tight_layout()
plt.show()
