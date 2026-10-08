import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Read image
# -----------------------------
img = cv2.imread("img_3.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

img = img.astype(np.float32) / 255.0


# -----------------------------
# 1. Gaussian Noise
# -----------------------------
def gaussian_noise(image, mean=0, sigma=0.1):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)


# -----------------------------
# 2. Salt & Pepper Noise
# -----------------------------
def salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()

    # Salt
    salt = np.random.random(image.shape) < amount / 2
    noisy[salt] = 1

    # Pepper
    pepper = np.random.random(image.shape) < amount / 2
    noisy[pepper] = 0

    return noisy


# -----------------------------
# 3. Speckle Noise
# -----------------------------
def speckle_noise(image, sigma=0.2):
    noise = np.random.randn(*image.shape) * sigma
    noisy = image + image * noise
    return np.clip(noisy, 0, 1)


# -----------------------------
# 4. Poisson Noise
# -----------------------------
def poisson_noise(image):
    vals = 2**8
    noisy = np.random.poisson(image * vals) / vals
    return np.clip(noisy, 0, 1)


# -----------------------------
# 5. Uniform Noise
# -----------------------------
def uniform_noise(image, low=-0.1, high=0.1):
    noise = np.random.uniform(low, high, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)


# -----------------------------
# 6. Rayleigh Noise
# -----------------------------
def rayleigh_noise(image, scale=0.1):
    noise = np.random.rayleigh(scale, image.shape)

    # Center the noise around zero
    noise = noise - np.mean(noise)

    noisy = image + noise
    return np.clip(noisy, 0, 1)


# -----------------------------
# Apply all 6 noises
# -----------------------------
gaussian = gaussian_noise(img)
salt_pepper = salt_pepper_noise(img)
speckle = speckle_noise(img)
poisson = poisson_noise(img)
uniform = uniform_noise(img)
rayleigh = rayleigh_noise(img)


# -----------------------------
# Display results
# -----------------------------
images = [img, gaussian, salt_pepper, speckle, poisson, uniform, rayleigh]

titles = [
    "Original Image",
    "Gaussian Noise",
    "Salt & Pepper Noise",
    "Speckle Noise",
    "Poisson Noise",
    "Uniform Noise",
    "Rayleigh Noise",
]

plt.figure(figsize=(12, 8))

for i in range(7):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
