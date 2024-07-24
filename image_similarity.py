import os
import signal
import argparse
import sqlite3
import tqdm
import torch
import threading
import sys
from queue import Queue
from PIL import Image, ExifTags
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

# Global variables for synchronization and signal handling
feature_queue = Queue()
abort_flag = threading.Event()

# Function to handle signal for aborting the process
def signal_handler(sig, frame):
    print('Aborted by user!')
    abort_flag.set()  # Set abort flag to stop processing

# Register signal handler for interrupt signal
signal.signal(signal.SIGINT, signal_handler)

def listen_for_abort():
    while not abort_flag.is_set():
        if sys.stdin.read(1) == 'q':
            abort_flag.set()  # Set abort flag to stop processing
    print("Operation aborted")

def if_exif_exists_reset(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
    except (AttributeError, KeyError, IndexError):
        pass

def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image don't have getexif
        pass
    return image

def get_clip_features(image_path):
    image = Image.open(image_path)
    image = correct_image_orientation(image)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().flatten()

def collect_image_paths(folder_path, existing_images):
    image_paths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if abort_flag.is_set():
                return image_paths
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dirpath, filename)
                if image_path not in existing_images:
                    image_paths.append(image_path)
    return image_paths

def process_images(image_paths, feature_queue, existing_images, process_progress):
    for image_path in image_paths:
        if abort_flag.is_set():
            feature_queue.put(None)  # Signal that processing is done
            return
        try:
            features = get_clip_features(image_path)
            feature_queue.put((image_path, features))
            process_progress.update(1)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    feature_queue.put(None)  # Signal that processing is done
    print("Completed processing of images")

def write_to_database(feature_queue, db_filename, write_progress):
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (filename TEXT PRIMARY KEY, features BLOB)''')

    while True:
        item = feature_queue.get()
        if item is None:
            break
        image_path, features = item
        c.execute("INSERT OR REPLACE INTO images VALUES (?, ?)", (image_path.replace('\\', '/'), features.tobytes()))
        write_progress.update(1)
        feature_queue.task_done()

    conn.commit()
    conn.close()
    print("Writing complete, connection closed")

def create_database(folder_name):
    print(f"Folder is {folder_name}")
    global feature_queue  # Ensure global queue is used
    db_filename = 'image_similarity.db'

    # SQLite database initialization
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (filename TEXT PRIMARY KEY, features BLOB)''')
    c.execute('SELECT filename FROM images')
    existing_images = {row[0] for row in c.fetchall()}
    conn.close()

    print(f"Total processed images in database: {len(existing_images)}")

    # Count total images in the directory
    image_paths = collect_image_paths(folder_name, existing_images)
    total_images_in_directory = len(image_paths)
    print(f"Total new images to process: {total_images_in_directory}")

    # Initialize progress bars
    process_progress = tqdm.tqdm(total=total_images_in_directory, desc="Processing images")
    write_progress = tqdm.tqdm(total=total_images_in_directory, desc="Writing to database")

    # Start processing and writing threads
    process_thread = threading.Thread(target=process_images, args=(image_paths, feature_queue, existing_images, process_progress))
    write_thread = threading.Thread(target=write_to_database, args=(feature_queue, db_filename, write_progress))

    process_thread.start()
    write_thread.start()

    # Start a thread to listen for 'q' keypresses
    abort_thread = threading.Thread(target=listen_for_abort, daemon=True)
    abort_thread.start()

    # Wait for all threads to finish
    process_thread.join()
    write_thread.join()

    print(f"Database updated: {db_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create image similarity database using CLIP ViT-B/16 model.')
    parser.add_argument('folder_name', type=str, nargs='?', default='F:/', help='Folder name within the default directory to scan for images')
    args = parser.parse_args()

    print("Press 'q' to abort...")    

    folder_name = args.folder_name
    create_database(folder_name)
    print("Closed main")