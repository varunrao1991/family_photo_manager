import os
import argparse
import sqlite3
import tqdm
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip  # Importing CLIP from the OpenAI repository

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# CLIP preprocessing pipeline
transform = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711]),
])

def get_clip_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy()[0]

def create_database(folder_name):
    root_dir = os.getcwd()  # Root directory where script is executed
    db_filename = os.path.join(root_dir, 'image_similarity.db')

    # SQLite database initialization
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (filename TEXT PRIMARY KEY, features TEXT)''')

    # Get existing image paths from the database
    c.execute('SELECT filename FROM images')
    existing_images = {row[0] for row in c.fetchall()}

    # Traverse directory recursively
    folder_path = os.path.join(root_dir, folder_name)
    image_paths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(dirpath, filename))

    total_images = len(image_paths)
    print(f"Total images found: {total_images}")

    # Insert images into database with progress bar
    with tqdm.tqdm(total=total_images, desc="Processing images") as pbar:
        for image_path in image_paths:
            if image_path not in existing_images:
                try:
                    features = get_clip_features(image_path)
                    features_str = ','.join(map(str, features))
                    c.execute("INSERT OR REPLACE INTO images VALUES (?, ?)", (image_path, features_str))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            pbar.update(1)

    conn.commit()
    conn.close()
    print(f"Database created: {db_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create image similarity database using OpenAI CLIP model.')
    parser.add_argument('folder_name', type=str, help='Folder name within current directory to scan for images')
    args = parser.parse_args()

    folder_name = args.folder_name
    create_database(folder_name)
