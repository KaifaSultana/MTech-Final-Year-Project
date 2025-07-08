import numpy as np
from skimage import metrics
import cv2

# ----------------------------
# Parameters
# ----------------------------

resized_width = 256
resized_height = 256

# ----------------------------
# Function to Calculate MSE and PSNR
# ----------------------------

def evaluate_quality(image1_path, image2_path, label=""):
    # Read both images as grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Failed to load images for {label}.")
        return

    # Resize both to common dimensions
    img1 = cv2.resize(img1, (resized_width, resized_height))
    img2 = cv2.resize(img2, (resized_width, resized_height))

    # Calculate MSE
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = metrics.peak_signal_noise_ratio(img1, img2, data_range=255)

    # Print Results
    print(f"\n[{label}]")
    print(f"MSE  : {mse:.4f}")
    if psnr == float('inf'):
        print("PSNR : ∞ (Images are identical)")
    else:
        print(f"PSNR : {psnr:.2f} dB")

# ----------------------------
# Paths
# ----------------------------

Original_Image_Path = r"D:\Python practice\Proposed Method\resized_outputs\resized_Baboon.png"
Recovery_Image_Path1 = r"D:\Python practice\Maurya\Final_Decrypt_Image.png"
Recovery_Image_Path2 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\recovered_secret.png"

Original_Cover1 = r"C:\Users\KAIFA\Pictures\Cover 2.jpg"
Original_Cover2 = r"C:\Users\KAIFA\Pictures\Cover 1.webp"
Original_Cover3 = r"C:\Users\KAIFA\Pictures\Cover 3.jpeg"

Baboon_Share1 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Baboon\resized_Baboon_share_1.png"
Baboon_Share2 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Baboon\resized_Baboon_share_2.png"
Baboon_Share3 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Baboon\resized_Baboon_share_3.png"

Lena_Share1 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_bit-256-x-256-Grayscale-Lena-Image\resized_bit-256-x-256-Grayscale-Lena-Image_share_1.png"
Lena_Share2 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_bit-256-x-256-Grayscale-Lena-Image\resized_bit-256-x-256-Grayscale-Lena-Image_share_2.png"
Lena_Share3 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_bit-256-x-256-Grayscale-Lena-Image\resized_bit-256-x-256-Grayscale-Lena-Image_share_3.png"
Cameraman_Share1 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Cameraman\resized_Cameraman_share_1.png"
Cameraman_Share2 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Cameraman\resized_Cameraman_share_2.png"
Cameraman_Share3 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_Cameraman\resized_Cameraman_share_3.png"

Peppers_Share1 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_peppers\resized_peppers_share_1.png"
Peppers_Share2 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_peppers\resized_peppers_share_2.png"
Peppers_Share3 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_peppers\resized_peppers_share_3.png"

X_Ray_Share1 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_X-Ray\resized_X-Ray_share_1.png"
X_Ray_Share2 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_X-Ray\resized_X-Ray_share_2.png"
X_Ray_Share3 = r"D:\Python practice\Proposed Method\Colour_LSB_Chaotic_shares\resized_X-Ray\resized_X-Ray_share_3.png"

evaluate_quality(Original_Image_Path, Recovery_Image_Path1, "Hybrid Decryption Output vs Original Secret")
evaluate_quality(Original_Image_Path, Recovery_Image_Path2, "Only LSB Recovery vs Original Secret")

evaluate_quality(Original_Cover1, Baboon_Share1, "Cover 1 vs Baboon Share 1")
evaluate_quality(Original_Cover2, Baboon_Share2, "Cover 2 vs Baboon Share 2")
evaluate_quality(Original_Cover3, Baboon_Share3, "Cover 3 vs Baboon Share 3")
evaluate_quality(Original_Cover1,Lena_Share1, "Cover 1 vs Lena Share 1")
evaluate_quality(Original_Cover2,Lena_Share2, "Cover 2 vs Lena Share 2")
evaluate_quality(Original_Cover3,Lena_Share3, "Cover 3 vs Lena Share 3")
evaluate_quality(Original_Cover1,Cameraman_Share1, "Cover 1 vs Cameraman Share 1")
evaluate_quality(Original_Cover2,Cameraman_Share2, "Cover 2 vs Cameraman Share 2")
evaluate_quality(Original_Cover3,Cameraman_Share3, "Cover 3 vs Cameraman Share 3")
evaluate_quality(Original_Cover1,Peppers_Share1, "Cover 1 vs Peppers Share 1")
evaluate_quality(Original_Cover2,Peppers_Share2, "Cover 2 vs Peppers Share 2")
evaluate_quality(Original_Cover3,Peppers_Share3, "Cover 3 vs Peppers Share 3")
evaluate_quality(Original_Cover1,X_Ray_Share1, "Cover 1 vs X_Ray Share 1")
evaluate_quality(Original_Cover2,X_Ray_Share2, "Cover 2 vs X_Ray Share 2")
evaluate_quality(Original_Cover3,X_Ray_Share3, "Cover 3 vs X_Ray Share 3")