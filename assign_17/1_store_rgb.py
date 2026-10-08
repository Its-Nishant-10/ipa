from PIL import Image
import numpy as np

image = Image.open("img_2.jpeg").convert("RGB")
img_array = np.array(image)
R = img_array[:, :, 0]
G = img_array[:, :, 1]
B = img_array[:, :, 2]
print("Red Matrix:")
print(R)
print("\nGreen Matrix:")
print(G)
print("\nBlue Matrix:")
print(B)
print("\nImage Size:", image.size)
