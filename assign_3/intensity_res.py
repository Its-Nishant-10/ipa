import cv2

input_file = input("Enter input image: ")
output_file = input("Enter output image: ")
bits = int(input("Enter intensity resolution (bits): "))
img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
levels = 2**bits
step = 256 // levels
result = (img // step) * step
cv2.imwrite(output_file, result)
print("Image saved successfully.")
