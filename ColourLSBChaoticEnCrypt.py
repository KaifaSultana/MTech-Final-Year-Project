import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

# ----------------------------
# Parameters
# ----------------------------

secret_paths = [
    r"D:\Python practice\Proposed Method\resized_outputs\resized_Baboon.png",
    r"D:\Python practice\Proposed Method\resized_outputs\resized_bit-256-x-256-Grayscale-Lena-Image.png",
    r"D:\Python practice\Proposed Method\resized_outputs\resized_Cameraman.png",
    r"D:\Python practice\Proposed Method\resized_outputs\resized_peppers.png",
    r"D:\Python practice\Proposed Method\resized_outputs\resized_X-Ray.jpeg"
]

cover_paths = [
    r"C:\Users\KAIFA\Pictures\Cover 2.jpg",
    r"C:\Users\KAIFA\Pictures\Cover 1.webp",
    r"C:\Users\KAIFA\Pictures\Cover 3.jpeg"
]

output_dir = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares"
os.makedirs(output_dir, exist_ok=True)

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

def embed_secret_in_lsb(secret, covers, seed=0.1234, r=3.999):
    m, n = secret.shape
    pixel_order = generate_chaotic_pixel_order(secret.shape, seed, r)
    shares = [cover.copy() for cover in covers]
    
    secret_bits = ''.join([format(pixel, '08b') for row in secret for pixel in row])
    total_bits = len(secret_bits)
    
    bit_idx = 0
    random.seed(42)

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
                old_val = shares[share_idx][i, j, channel]
                new_val = (old_val & 0xFE) | int(secret_bits[bit_idx])
                shares[share_idx][i, j, channel] = new_val
                bit_idx += 1
    return shares

# ----------------------------
# Main Execution
# ----------------------------

for secret_path in secret_paths:
    secret_img = Image.open(secret_path).convert('L')
    secret_arr = np.array(secret_img)
    height, width = secret_arr.shape

    # Load and resize covers to match secret image size
    cover_images = [
        np.array(Image.open(p).convert('RGB').resize((width, height)))
        for p in cover_paths
    ]

    shares_arr = embed_secret_in_lsb(secret_arr, cover_images)

    base_name = os.path.splitext(os.path.basename(secret_path))[0]
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)

    # Save shares
    for idx, arr in enumerate(shares_arr):
        Image.fromarray(arr).save(os.path.join(output_subdir, f'{base_name}_share_{idx+1}.png'))

    # Show result
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    axs[0].imshow(secret_arr, cmap='gray')
    axs[0].set_title(f'Secret: {base_name}')
    for i in range(3):
        axs[i+1].imshow(shares_arr[i])
        axs[i+1].set_title(f'Share {i+1}')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
