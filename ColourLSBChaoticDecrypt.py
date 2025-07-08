import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

# ----------------------------
# Parameters
# ----------------------------

share_paths = [
    # r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\share_1.png",
    # r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\share_2.png",
    # r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\share_3.png"
    r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Baboon\resized_Baboon_share_1.png",
    r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Baboon\resized_Baboon_share_2.png",
    r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Baboon\resized_Baboon_share_3.png"
]

output_path = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\recovered_secret_Baboon.png"
secret_shape = (256, 256)  # (height, width)
seed = 0.1234
r = 3.999

# ----------------------------
# Helper Functions
# ----------------------------

def logistic_map(x0, r, size):
    x = x0
    seq = []
    for _ in range(size):
        x = r * x * (1 - x)
        seq.append(x)
    return np.array(seq)

def generate_chaotic_pixel_order(shape, seed=0.1234, r=3.999):
    m, n = shape
    total_pixels = m * n
    chaotic_seq = logistic_map(seed, r, total_pixels)
    indices = np.argsort(-chaotic_seq)
    pixel_order = [(idx // n, idx % n) for idx in indices]
    return pixel_order

def extract_secret_from_lsb(shares, secret_shape, seed=0.1234, r=3.999):
    m, n = secret_shape
    pixel_order = generate_chaotic_pixel_order(secret_shape, seed, r)
    total_bits = m * n * 8
    secret_bits = []

    bit_idx = 0
    random.seed(42)  # Must match embedding

    for i, j in pixel_order:
        if bit_idx >= total_bits:
            break

        share_indices = [0, 1, 2]
        channel_indices = [0, 1, 2]
        random.shuffle(share_indices)
        random.shuffle(channel_indices)

        for share_idx in share_indices:
            if bit_idx >= total_bits:
                break
            for channel in channel_indices:
                if bit_idx >= total_bits:
                    break
                bit = shares[share_idx][i, j, channel] & 1
                secret_bits.append(str(bit))
                bit_idx += 1

    # Reconstruct grayscale image
    pixels = [int(''.join(secret_bits[i:i+8]), 2) for i in range(0, total_bits, 8)]
    secret_arr = np.array(pixels, dtype=np.uint8).reshape((m, n))
    return secret_arr

# ----------------------------
# Load Share Images
# ----------------------------

share_imgs = [Image.open(path).convert('RGB') for path in share_paths]
share_arrs = [np.array(img) for img in share_imgs]

# ----------------------------
# Recover Secret Image
# ----------------------------

recovered_secret = extract_secret_from_lsb(share_arrs, secret_shape, seed=seed, r=r)
Image.fromarray(recovered_secret).save(output_path)

# ----------------------------
# Display
# ----------------------------

plt.figure(figsize=(5, 5))
plt.imshow(recovered_secret, cmap='gray')
plt.title("Recovered Secret Image")
plt.axis('off')
plt.tight_layout()
plt.show()
