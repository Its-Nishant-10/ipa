from PIL import Image
import numpy as np

image = Image.open("img_2.jpeg").convert("L")
img = np.array(image)
low = 100
high = 200
sliced = np.zeros_like(img)
sliced[(img >= low) & (img <= high)] = 255
output = Image.fromarray(sliced)
output.save("intensity_sliced.jpg")
image.show()
output.show()
print("Intensity slicing completed.")
print("Selected intensity range:", low, "to", high)
