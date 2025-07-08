import numpy as np
from PIL import Image
import hashlib
import cv2
import math

# Paths
plain_image_path = r"C:\Users\KAIFA\Pictures\bit-256-x-256-Grayscale-Lena-Image.png"
key_image_path = "key_image.png"
revealed_share_image_path = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\recovered_secret.png"
p = 0.3
mu = 3.99999999

def load_image(filepath):
    """Loads an image in grayscale mode."""
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Could not load image from {filepath}")
    print(f"Loaded image '{filepath}' with shape {image.shape} and dtype {image.dtype}")
    return image

def save_image(image, filepath):
    """Saves a numpy array as an image."""
    Image.fromarray(image).save(filepath)

def rotate_image(image):
    """Rotates the image by 90 degrees anticlockwise."""
    return np.rot90(image, k=1)

def rotate_image_clockwise(image):
    """Rotates the image by 90 degrees clockwise."""
    return np.rot90(image, k=-1)

def generate_x0(image_path):
    """Generate the initial value x0 for a chaotic map using the first 32 bits of an MD5 hash."""
    with open(image_path, 'rb') as f:
        md5_hash = hashlib.md5(f.read()).digest()  # Get 16-byte binary hash
    d1, d2, d3, d4 = md5_hash[:4]  # Each is a single byte (8 bits)
    print(f"d1: {d1} ({hex(d1)})")
    print(f"d2: {d2} ({hex(d2)})")
    print(f"d3: {d3} ({hex(d3)})")
    print(f"d4: {d4} ({hex(d4)})")
    x0 = ((d1 ^ d2 ^ d3 ^ d4) % 256) / 255  # Equation 3
    if x0 == 0 or x0 == 1:
        raise ValueError("Generated x0 is 0 or 1. Repeat the process using Eq. (3).")
    return x0

def chaotic_map(x, p):
    if 0 < x < p:
        return x / p
    elif p <= x < 0.5:
        return (x - p) / (0.5 - p)
    elif 0.5 <= x < 1:
        return chaotic_map(1 - x, p)

def generate_chaotic_sequence(length, p, x0):
    """Generates a chaotic sequence using Equation 1."""
    sequence = []
    x = x0
    for _ in range(length):
        x = chaotic_map(x, p)
        sequence.append(x)
    return np.array(sequence)

def generate_key_image(plain_image, chaotic_sequence):
    """Generates the key image using chaotic sequences."""
    rows, cols = plain_image.shape
    key_pixels = np.floor((chaotic_sequence * 256)).astype(np.uint8)  # Apply Equation 4
    scaled_sequence = np.floor(chaotic_sequence * 256)
    for i in range(min(3, len(chaotic_sequence))):
        print(f"chaotic_sequence[{i}] = {chaotic_sequence[i]}")
        print(f"key_pixel = floor({chaotic_sequence[i]} * 256) = floor({chaotic_sequence[i] * 256}) = {scaled_sequence[i]}\n")
    key_image = key_pixels.reshape((rows, cols))
    print("Generated Key Image Array:")
    print(key_image)
    save_image(key_image, key_image_path)
    return key_image

def logistic_map(x0, mu, n_iter):
    """Generates a sequence using the logistic map (Equation (2))."""
    sequence = [x0]
    x = x0
    for _ in range(n_iter):
        x = mu * x * (1 - x)
        sequence.append(x)
    return sequence

def decide_rna_rule(sequence):
    """Decides RNA rule based on Equation (5): Rule = floor(x * 8) + 1."""
    return [int(np.floor(x * 8)) + 1 for x in sequence]

# RNA Rules Table (replacing T with U)
rna_rules_table = {
    'Rule1': {'00': 'A', '01': 'C', '10': 'G', '11': 'U'},
    'Rule2': {'00': 'A', '01': 'G', '10': 'C', '11': 'U'},
    'Rule3': {'00': 'U', '01': 'C', '10': 'G', '11': 'A'},
    'Rule4': {'00': 'U', '01': 'G', '10': 'C', '11': 'A'},
    'Rule5': {'00': 'C', '01': 'A', '10': 'U', '11': 'G'},
    'Rule6': {'00': 'C', '01': 'U', '10': 'A', '11': 'G'},
    'Rule7': {'00': 'G', '01': 'A', '10': 'U', '11': 'C'},
    'Rule8': {'00': 'G', '01': 'U', '10': 'A', '11': 'C'}
}

# RNA Operation Tables (replacing T with U)
xor_table = {
    'A': {'A': 'A', 'C': 'C', 'U': 'U', 'G': 'G'},
    'U': {'A': 'U', 'C': 'G', 'U': 'A', 'G': 'C'},
    'C': {'A': 'C', 'C': 'A', 'U': 'G', 'G': 'U'},
    'G': {'A': 'G', 'C': 'U', 'U': 'C', 'G': 'A'}
}

add_table = {
    'A': {'A': 'C', 'C': 'A', 'U': 'G', 'G': 'U'},
    'U': {'A': 'G', 'C': 'U', 'U': 'C', 'G': 'A'},
    'C': {'A': 'A', 'C': 'C', 'U': 'U', 'G': 'G'},
    'G': {'A': 'U', 'C': 'G', 'U': 'A', 'G': 'C'}
}

sub_table = {
    'A': {'A': 'C', 'C': 'G', 'U': 'A', 'G': 'U'},
    'U': {'A': 'G', 'C': 'U', 'U': 'C', 'G': 'A'},
    'C': {'A': 'A', 'C': 'C', 'U': 'U', 'G': 'G'},
    'G': {'A': 'U', 'C': 'A', 'U': 'G', 'G': 'C'}
}

reverse_sub_table = {
    'A': {'A': 'C', 'C': 'G', 'U': 'A', 'G': 'U'},
    'U': {'A': 'G', 'C': 'U', 'U': 'C', 'G': 'A'},
    'C': {'A': 'A', 'C': 'C', 'U': 'U', 'G': 'G'},
    'G': {'A': 'U', 'C': 'A', 'U': 'G', 'G': 'C'}
}

def rna_encode_image(image, sequence, rules_table):
    """Encodes an image into RNA using row-specific rules."""
    M, N = image.shape
    encoded_image = np.zeros((4, M, N), dtype='<U1')
    rules = decide_rna_rule(sequence)
    max_rows_to_print = 3
    max_cols_to_print = 3
    for row in range(M):
        rule_name = f"Rule{rules[row]}"
        sequence_value = sequence[row]
        for col in range(N):
            pixel = image[row, col]
            binary_pixel = f"{pixel:08b}"
            rna_bases = [rules_table[rule_name][binary_pixel[i:i+2]] for i in range(0, 8, 2)]
            encoded_image[:, row, col] = rna_bases
            if row < max_rows_to_print and col < max_cols_to_print:
                print(f"Row {row}, Col {col}: Pixel = {pixel}, Binary = {binary_pixel}, RNA Bases = {rna_bases}, Sequence Value = {sequence_value}, Rule = {rule_name}")
    return encoded_image

def rna_decode_image(encoded_image, sequence, rules_table):
    """Decodes an RNA-encoded image back to grayscale."""
    M, N = encoded_image.shape[1:]
    decoded_image = np.zeros((M, N), dtype=np.uint8)
    rules = decide_rna_rule(sequence)
    max_rows_to_print = 3
    max_cols_to_print = 3
    for row in range(M):
        rule_name = f"Rule{rules[row]}"
        reverse_rule = {v: k for k, v in rules_table[rule_name].items()}
        for col in range(N):
            rna_bases = encoded_image[:, row, col]
            binary_pixel = ''.join(reverse_rule[base] for base in rna_bases)
            decoded_image[row, col] = int(binary_pixel, 2)
            if row < max_rows_to_print and col < max_cols_to_print:
                print(f"Row {row}, Col {col}: RNA Bases = {rna_bases}, Binary = {binary_pixel}, Decoded Pixel = {decoded_image[row, col]}, Rule = {rule_name}")
    return decoded_image

def rna_operations(encoded_plain, encoded_key, sequence, xor_table, add_table, sub_table):
    """Performs RNA operations row by row."""
    rows, cols = encoded_plain.shape[1:]
    result_image = np.zeros_like(encoded_plain)
    max_rows_to_print = 10
    max_cols_to_print = 10
    for row in range(rows):
        operation = math.floor(sequence[row] * 3) + 1
        for col in range(cols):
            plain_rna = encoded_plain[:, row, col]
            key_rna = encoded_key[:, row, col]
            if operation == 1:  # XOR
                result_image[:, row, col] = [xor_table[p][k] for p, k in zip(plain_rna, key_rna)]
                op_type = "XOR"
            elif operation == 2:  # Addition
                result_image[:, row, col] = [add_table[p][k] for p, k in zip(plain_rna, key_rna)]
                op_type = "Addition"
            elif operation == 3:  # Subtraction
                result_image[:, row, col] = [sub_table[p][k] for p, k in zip(plain_rna, key_rna)]
                op_type = "Subtraction"
            if row < max_rows_to_print and col < max_cols_to_print:
                print(f"Row {row}, Col {col}: Plain RNA = {plain_rna}, Key RNA = {key_rna}, Operation = {op_type}")
                print(f"Resulting RNA: {result_image[:, row, col]}")
    return result_image

# Main Execution
plain_image = load_image(plain_image_path)
rows, cols = plain_image.shape
print(f"Value of plain image:\n{plain_image[:4,:4]}")
x0 = generate_x0(plain_image_path)
chaotic_sequence = generate_chaotic_sequence(rows * cols, p, x0)
print("Generated Chaotic Sequence:")
print(chaotic_sequence)
key_image = generate_key_image(plain_image, chaotic_sequence)
print(f"Pixel value of key Image:\n{key_image[:4, :4]}")
logistic_sequence = logistic_map(x0, mu, plain_image.shape[0])

# RNA Encoding and Operations
encoded_image = rna_encode_image(plain_image, logistic_sequence, rna_rules_table)
rna_key = rna_encode_image(key_image, logistic_sequence, rna_rules_table)
intermediate_image = rna_operations(encoded_image, rna_key, logistic_sequence, xor_table, add_table, sub_table)
decoded_intermediate_image = rna_decode_image(intermediate_image, logistic_sequence, rna_rules_table)
rotated_intermediate_image = rotate_image(decoded_intermediate_image)
rotate_encoded = rna_encode_image(rotated_intermediate_image, logistic_sequence, rna_rules_table)
rotated_encoded_operation = rna_operations(rotate_encoded, rna_key, logistic_sequence, xor_table, add_table, sub_table)
decoded_rotated_encoded_operation = rna_decode_image(rotated_encoded_operation, logistic_sequence, rna_rules_table)
save_image(decoded_rotated_encoded_operation, "dencoded_Intermediate_image.png")  # Scrambled image
#Decryption
reverse_encode = rna_encode_image(decoded_rotated_encoded_operation, logistic_sequence, rna_rules_table)
reverse_operation = rna_operations(reverse_encode, rna_key, logistic_sequence, xor_table, add_table, reverse_sub_table)
reverse_decode = rna_decode_image(reverse_operation, logistic_sequence, rna_rules_table)
reverse_rotate_image = rotate_image_clockwise(reverse_decode)
encode_intermediate_image = rna_encode_image(reverse_rotate_image, logistic_sequence, rna_rules_table)
intermediate_image_reverse = rna_operations(encode_intermediate_image, rna_key, logistic_sequence, xor_table, add_table, reverse_sub_table)
final_decrypted_image = rna_decode_image(intermediate_image_reverse, logistic_sequence, rna_rules_table)
save_image(final_decrypted_image, "Final_Decrypt_Image.png")