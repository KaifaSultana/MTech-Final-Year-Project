import numpy as np
from skimage import metrics
import cv2

# Resize images to a common size
resized_width = 256
resized_height = 256

# Function to calculate PSNR for grayscale images
def psnr_calculate(image1, image2):
    # Read the images in grayscale mode
    fimage1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    fimage2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded successfully
    if fimage1 is None or fimage2 is None:
        print("Failed to load one or both images.")
        return None
    
    # Resize images to a common size
    fimage1 = cv2.resize(fimage1, (resized_width, resized_height))
    fimage2 = cv2.resize(fimage2, (resized_width, resized_height))

    # Compute Mean Squared Error
    mse = np.mean((fimage1 - fimage2) ** 2)
    print(f"mse :{mse}")
    if mse == 0:
        print("PSNR Value: ∞ (images are identical)")
        return float('inf')  # Images are identical
    else:
        # Compute PSNR
        psnr = metrics.peak_signal_noise_ratio(fimage1, fimage2, data_range=255)
        print(f"PSNR Value: {psnr:.2f} dB")
        return psnr

# Example Usage
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
psnr_value1 = psnr_calculate(Original_Image_Path, Recovery_Image_Path1)
psnr_value2 = psnr_calculate(Original_Image_Path, Recovery_Image_Path2)
psnr_value3 = psnr_calculate(Original_Cover1,Baboon_Share1)
psnr_value4 = psnr_calculate(Original_Cover2,Baboon_Share2)
psnr_value5 = psnr_calculate(Original_Cover3,Baboon_Share3)
print(f"PSNR Value of hybrid method: {psnr_value1}")
print(f"PSNR Value of only DNA Encrypted Image: {psnr_value2}")
print(f"PSNR value between Cover_1 and Baboon_Share1: {psnr_value3}")
# print(f"PSNR Value3: {psnr_value3}")