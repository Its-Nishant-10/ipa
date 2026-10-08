import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# READ IMAGE
# ==========================================
img = cv2.imread("img_1.jpeg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# Convert image to 0-1 range
img = img.astype(np.float32) / 255.0


# ==========================================
# 1. GAUSSIAN NOISE
# ==========================================
def gaussian_noise(image, mean=0, sigma=0.1):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)


# ==========================================
# 2. SALT AND PEPPER NOISE
# ==========================================
def salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()

    salt = np.random.random(image.shape) < amount / 2
    pepper = np.random.random(image.shape) < amount / 2

    noisy[salt] = 1
    noisy[pepper] = 0

    return noisy


# ==========================================
# 3. SPECKLE NOISE
# ==========================================
def speckle_noise(image, sigma=0.2):
    noise = np.random.randn(*image.shape) * sigma
    noisy = image + image * noise

    return np.clip(noisy, 0, 1)


# ==========================================
# 4. POISSON NOISE
# ==========================================
def poisson_noise(image):
    vals = 256

    noisy = np.random.poisson(image * vals) / vals

    return np.clip(noisy, 0, 1)


# ==========================================
# 5. UNIFORM NOISE
# ==========================================
def uniform_noise(image, low=-0.1, high=0.1):
    noise = np.random.uniform(low, high, image.shape)

    noisy = image + noise

    return np.clip(noisy, 0, 1)


# ==========================================
# 6. RAYLEIGH NOISE
# ==========================================
def rayleigh_noise(image, scale=0.1):
    noise = np.random.rayleigh(scale, image.shape)

    # Center noise around zero
    noise = noise - np.mean(noise)

    noisy = image + noise

    return np.clip(noisy, 0, 1)


# ==========================================
# APPLY ALL 6 NOISES
# ==========================================
gaussian = gaussian_noise(img)

salt_pepper = salt_pepper_noise(img)

speckle = speckle_noise(img)

poisson = poisson_noise(img)

uniform = uniform_noise(img)

rayleigh = rayleigh_noise(img)


# ==========================================
# SNR CALCULATION
# ==========================================
def calculate_snr(original, noisy):

    # Signal power
    signal_power = np.mean(original**2)

    # Noise
    noise = noisy - original

    # Noise power
    noise_power = np.mean(noise**2)

    # Avoid division by zero
    if noise_power == 0:
        return float("inf")

    # SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)

    return snr


# ==========================================
# CALCULATE SNR
# ==========================================
snr_gaussian = calculate_snr(img, gaussian)
snr_sp = calculate_snr(img, salt_pepper)
snr_speckle = calculate_snr(img, speckle)
snr_poisson = calculate_snr(img, poisson)
snr_uniform = calculate_snr(img, uniform)
snr_rayleigh = calculate_snr(img, rayleigh)


# ==========================================
# PRINT SNR VALUES
# ==========================================
print("--------------------------------------")
print("          SNR RESULTS")
print("--------------------------------------")

print("Original Image SNR : Infinity dB")
print("Gaussian Noise SNR : {:.2f} dB".format(snr_gaussian))
print("Salt & Pepper SNR  : {:.2f} dB".format(snr_sp))
print("Speckle Noise SNR  : {:.2f} dB".format(snr_speckle))
print("Poisson Noise SNR  : {:.2f} dB".format(snr_poisson))
print("Uniform Noise SNR  : {:.2f} dB".format(snr_uniform))
print("Rayleigh Noise SNR : {:.2f} dB".format(snr_rayleigh))


# ==========================================
# DISPLAY IMAGES
# ==========================================
images = [img, gaussian, salt_pepper, speckle, poisson, uniform, rayleigh]

titles = [
    "Original",
    "Gaussian Noise",
    "Salt & Pepper",
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
