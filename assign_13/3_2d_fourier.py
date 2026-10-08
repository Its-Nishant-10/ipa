import cv2
import numpy as np
import matplotlib.pyplot as plt

# Put your 4 image filenames here
image_files = ["img_1.jpeg", "img_2.jpeg", "img_3.jpeg"]

plt.figure(figsize=(12, 10))

for i, file in enumerate(image_files):

    # Read image in grayscale
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not read {file}")
        continue

    # 2D Fourier Transform
    f = np.fft.fft2(img)

    # Move low frequencies to the center
    f_shift = np.fft.fftshift(f)

    # Calculate magnitude spectrum
    magnitude = np.abs(f_shift)

    # Log transformation for better visualization
    magnitude_spectrum = np.log(1 + magnitude)

    # Display original image
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Original Image {i+1}")
    plt.axis("off")

    # Display Fourier transform
    plt.subplot(4, 2, 2 * i + 2)
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title(f"2D Fourier Transform {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
