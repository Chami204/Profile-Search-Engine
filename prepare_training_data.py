import os
from pdf2image import convert_from_path
import cv2
import numpy as np
from utils.image_utils import preprocess_drawing
from pathlib import Path
import random

DATA_DIR = r"C:\Users\chami.gangoda\OneDrive - Hayleys Group\Desktop\Software creations\CNN model for search engine\data"
OUT_DIR = r"C:\Users\chami.gangoda\OneDrive - Hayleys Group\Desktop\Software creations\CNN model for search engine\training_data"

# -------------------------------------------------------------------
# 1) Extract pages from PDFs
def extract_images_from_pdf(pdf_path, output_dir, max_pages=3):
    pages = convert_from_path(
        pdf_path,
        dpi=300,
        poppler_path=r"C:\Users\chami.gangoda\OneDrive - Hayleys Group\Desktop\Software creations\CNN model for search engine\Release-24.08.0-0 (1)\poppler-24.08.0\Library\bin"
    )
    saved_images = []
    for i, page in enumerate(pages[:max_pages]):
        img_path = os.path.join(output_dir, f"page_{i}.jpg")
        page.save(img_path, "JPEG")
        saved_images.append(img_path)
    return saved_images

# -------------------------------------------------------------------
# 2) Augment full-size cleaned images
def augment_image(image):
    augmented = []

    # Increase contrast without distorting fine details
    enhanced = cv2.convertScaleAbs(image, alpha=1.1, beta=5)  # Slight brightness boost

    # Apply slight Gaussian blur to prevent harsh feature loss
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Horizontal + vertical flips
    augmented.append(cv2.flip(blurred, 1))
    augmented.append(cv2.flip(blurred, 0))

    # Rotations while keeping feature preservation
    center = (blurred.shape[1] // 2, blurred.shape[0] // 2)
    for angle in [5, -5, 10, -10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(enhanced, M, (enhanced.shape[1], enhanced.shape[0]), flags=cv2.INTER_CUBIC, borderValue=int(image.mean()))

        # Use fine-tuned binary thresholding to preserve middle shapes
        _, rotated = cv2.threshold(rotated, 180, 255, cv2.THRESH_BINARY)
        augmented.append(rotated)

    return augmented

# -------------------------------------------------------------------
# 3) Prepare training data (extract, clean, save, augment)
def prepare_training_data():
    os.makedirs(OUT_DIR, exist_ok=True)

    for file in os.listdir(DATA_DIR):
        if not file.lower().endswith(".pdf"):
            continue

        profile_name = Path(file).stem
        pdf_path = os.path.join(DATA_DIR, file)
        profile_folder = os.path.join(OUT_DIR, profile_name)
        os.makedirs(profile_folder, exist_ok=True)

        # Extract pages
        raw_images = extract_images_from_pdf(pdf_path, profile_folder)

        for idx, raw_img_path in enumerate(raw_images):
            # Clean & crop to A4
            cleaned = preprocess_drawing(raw_img_path)

            # Save original A4-size cleaned drawing
            cv2.imwrite(os.path.join(profile_folder, f"a4_{idx}.jpg"), cleaned)

            # Duplicate for model input (same resolution)
            cv2.imwrite(os.path.join(profile_folder, f"cnn_{idx}.jpg"), cleaned)

            # Augment and save
            for j, aug in enumerate(augment_image(cleaned)):
                aug_path = os.path.join(profile_folder, f"aug_{idx}_{j}.jpg")
                cv2.imwrite(aug_path, aug)

# -------------------------------------------------------------------
# 4) Clean residual letters, numbers, dots from trained images
MIN_AREA = 300  # Reduced minimum area to preserve smaller structures
OPEN_KERNEL = np.ones((2, 2), np.uint8)  # Smaller kernel to avoid erasing useful details

def clean_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # Apply Gaussian blur to prevent harsh noise elimination
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Invert & threshold to white-on-black (fine-tuned value)
    _, bw = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)  # Increase threshold sensitivity

    # Remove tiny specks while retaining mid-sized structures
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, OPEN_KERNEL, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, OPEN_KERNEL, iterations=1)


    # Keep only large connected components (fine-tuned threshold)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    mask = np.zeros_like(bw)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            mask[labels == i] = 255

    # Invert back to black-on-white
    cleaned = cv2.bitwise_not(mask)
    cv2.imwrite(path, cleaned)


def clean_folder():
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                clean_image(os.path.join(root, f))
    print("✅ Cleaned all small letters, numbers, and dots from training images.")

# -------------------------------------------------------------------
if __name__ == "__main__":
    prepare_training_data()
    clean_folder()
    print("✅ Training data prep and post-cleanup complete.")
