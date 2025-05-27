from pdf2image import convert_from_path
import os

def convert_pdf_to_images(pdf_path, output_folder="C:\Users\chami.gangoda\OneDrive - Hayleys Group\Desktop\Software creations\CNN model for search engine\database"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pages = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, page in enumerate(pages):
        img_path = f"{output_folder}/page_{i}.jpg"
        page.save(img_path, "JPEG")
        image_paths.append(img_path)
    return image_paths
