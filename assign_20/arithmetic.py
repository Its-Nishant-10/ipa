from PIL import Image
from collections import Counter
from fractions import Fraction


def arithmetic_encode(data):
    freq = Counter(data)
    total = len(data)
    symbols = sorted(freq.keys())
    ranges = {}
    cumulative = 0
    for symbol in symbols:
        symbol_low = Fraction(cumulative, total)
        cumulative += freq[symbol]
        symbol_high = Fraction(cumulative, total)
        ranges[symbol] = (symbol_low, symbol_high)
    low = Fraction(0, 1)
    high = Fraction(1, 1)
    for symbol in data:
        symbol_low, symbol_high = ranges[symbol]
        current_range = high - low
        high = low + current_range * symbol_high
        low = low + current_range * symbol_low
    encoded_value = (low + high) / 2
    return encoded_value, freq, low, high
def arithmetic_decode(encoded_value, freq, data_length):
    total = sum(freq.values())
    symbols = sorted(freq.keys())
    ranges = {}
    cumulative = 0
    for symbol in symbols:
        symbol_low = Fraction(cumulative, total)
        cumulative += freq[symbol]
        symbol_high = Fraction(cumulative, total)
        ranges[symbol] = (symbol_low, symbol_high)
    low = Fraction(0, 1)
    high = Fraction(1, 1)
    decoded = []
    for _ in range(data_length):
        current_range = high - low
        value = (encoded_value - low) / current_range
        for symbol in symbols:
            symbol_low, symbol_high = ranges[symbol]
            if symbol_low <= value < symbol_high:
                decoded.append(symbol)
                high = low + current_range * symbol_high
                low = low + current_range * symbol_low
                break
    return decoded
input_file = "img_1.jpeg"
output_file = "fig_1.png"
image = Image.open(input_file).convert("L")
width, height = image.size
pixels = [pixel for pixel in image.getdata()]
encoded_value, frequency, _, _ = arithmetic_encode(pixels)

decoded_pixels = arithmetic_decode(encoded_value, frequency, len(pixels))

decoded_image = Image.new("L", (width, height))
decoded_image.putdata(decoded_pixels)
decoded_image.save(output_file)
