import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_1.jpeg", 0)
F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)
M, N = img.shape
u = np.arange(-N // 2, N // 2)
v = np.arange(-M // 2, M // 2)
U, V = np.meshgrid(u, v)
D = np.sqrt(U**2 + V**2)
D_low = 20
D_high = 80
n = 2
H_BP = 1 / (1 + (D_low / (D + 1e-5)) ** (2 * n)) * 1 / (1 + (D / D_high) ** (2 * n))
G_BP = F_shift * H_BP
result_BP = np.fft.ifft2(np.fft.ifftshift(G_BP))
result_BP = np.abs(result_BP)
result_BP = cv2.normalize(result_BP, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
plt.imshow(result_BP, cmap="gray")
plt.title("Butterworth Band-Pass Filter")
plt.axis("off")
plt.show()
H_BS = 1 - H_BP
G_BS = F_shift * H_BS
result_BS = np.fft.ifft2(np.fft.ifftshift(G_BS))
result_BS = np.abs(result_BS)
result_BS = cv2.normalize(result_BS, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
plt.imshow(result_BS, cmap="gray")
plt.title("Butterworth Band-Stop Filter")
plt.axis("off")
plt.show()
