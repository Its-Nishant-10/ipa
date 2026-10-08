from PIL import Image

input_file = input("Enter input image path: ")
output_file = input("Enter output image path: ")
dpi = int(input("Enter new DPI: "))
img = Image.open(input_file)
img.save(output_file, dpi=(dpi, dpi))
print(f"Image saved successfully with {dpi} DPI.")
