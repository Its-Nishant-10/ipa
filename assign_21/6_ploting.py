import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def matrix(n):
    x = np.arange(n)
    scale = np.full(n, np.sqrt(2 / n))
    scale[0] = np.sqrt(1 / n)
    return scale[:, None] * np.cos(np.pi * (2 * x + 1) * x[:, None] / (2 * n))


def scan(a):
    n = a.shape[0]
    return [
        a[i, s - i]
        for s in range(2 * n - 1)
        for i in (
            range(min(s, n - 1), max(-1, s - n), -1)
            if s % 2 == 0
            else range(max(0, s - n + 1), min(s, n - 1) + 1)
        )
    ]


def unscan(values, n):
    a = np.zeros((n, n), dtype=int)
    index = 0

    for s in range(2 * n - 1):
        rows = (
            range(min(s, n - 1), max(-1, s - n), -1)
            if s % 2 == 0
            else range(max(0, s - n + 1), min(s, n - 1) + 1)
        )

        for i in rows:
            a[i, s - i] = values[index]
            index += 1

    return a


def rle(values):
    result = []
    start = 0

    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] != values[start]:
            result.append((int(values[start]), end - start))
            start = end

    return result


def compress(image, size):
    h, w = image.shape
    h = h - h % size
    w = w - w % size

    output = np.zeros((h, w))
    q = 1 + np.indices((size, size)).sum(axis=0)
    d = matrix(size)
    encoded = 0

    for i in range(0, h, size):
        for j in range(0, w, size):
            block = image[i : i + size, j : j + size]

            values = np.round(d @ block @ d.T / q).astype(int)

            pairs = rle(scan(values))
            encoded += len(pairs)

            values = unscan(
                np.repeat([x for x, _ in pairs], [n for _, n in pairs]), size
            )

            output[i : i + size, j : j + size] = d.T @ (values * q) @ d

    return (np.clip(output, 0, 255).astype(np.uint8), image.size / max(encoded, 1))


def rms(a, b):
    return np.sqrt(np.mean((a.astype(float) - b.astype(float)) ** 2))


def dft(image):
    transformed = np.fft.fft2(image)

    count = max(1, int(transformed.size * 0.1))

    threshold = np.partition(np.abs(transformed).ravel(), -count)[-count]

    return np.clip(
        np.real(
            np.fft.ifft2(np.where(np.abs(transformed) >= threshold, transformed, 0))
        ),
        0,
        255,
    ).astype(np.uint8)


def main():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")

    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".png"))[:15]

    if not names:
        raise FileNotFoundError(f"No PNG images found in: {folder}")

    images = []

    for n in names:
        path = os.path.join(folder, n)

        try:
            image = Image.open(path)
            image.load()
            image = image.convert("L")
            image = image.resize((256, 256))
            images.append(np.asarray(image))
        except Exception:
            print("Skipped:", n)

    if not images:
        raise FileNotFoundError("No valid PNG images found.")

    print("Images used:")

    for i, n in enumerate(names[: len(images)], 1):
        print(i, n)

    sizes = (2, 4, 8)

    errors = {f"DCT {size}x{size}": [] for size in sizes}

    errors["DFT"] = []

    for image in images:
        for size in sizes:
            compressed, _ = compress(image, size)

            errors[f"DCT {size}x{size}"].append(rms(image, compressed))

        errors["DFT"].append(rms(image, dft(image)))

    x = np.arange(1, len(images) + 1)

    plt.figure(figsize=(10, 6))

    for label, values in errors.items():
        plt.plot(x, values, "o-", label=label)

    plt.xlabel("Image Number")
    plt.ylabel("RMS Error")
    plt.title("DFT vs DCT RMS Error")
    plt.legend()
    plt.grid()
    plt.show()

    original = images[0]

    dct2 = compress(original, 2)[0]
    dct4 = compress(original, 4)[0]
    dct8 = compress(original, 8)[0]

    results = [
        ("Original", original),
        ("DCT 2x2", dct2),
        ("DCT 4x4", dct4),
        ("DCT 8x8", dct8),
    ]

    plt.figure(figsize=(10, 8))

    for i, (title, result) in enumerate(results, 1):
        plt.subplot(2, 2, i)
        plt.imshow(result, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
