from PIL import Image
import os



def resize_images_with_prefix(image_paths, output_dir='resized_outputs'):
    os.makedirs(output_dir, exist_ok=True)

    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_resized = img.resize((256, 256))

        # Extract original filename
        filename = os.path.basename(path)
        new_filename = f"resized_{filename}"
        save_path = os.path.join(output_dir, new_filename)

        img_resized.save(save_path)
        print(f"Saved: {save_path}")

image_paths = [
    r"C:\Users\KAIFA\Pictures\Baboon.png",
    r"C:\Users\KAIFA\Pictures\bit-256-x-256-Grayscale-Lena-Image.png",
    r"C:\Users\KAIFA\Pictures\man.bmp",
    r"C:\Users\KAIFA\Pictures\Cameraman.png",
    r"C:\Users\KAIFA\Pictures\peppers.png",
    r"C:\Users\KAIFA\Pictures\X-Ray.jpeg"
    
]

resize_images_with_prefix(image_paths)

