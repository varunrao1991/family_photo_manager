import os
import sys
import signal
import argparse
import sqlite3
import threading
from queue import Queue
from tqdm import tqdm
from PIL import Image, ExifTags
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip

# Initialize CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

transform = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711]),
])

feature_queue = Queue()
abort_flag = threading.Event()


def signal_handler(sig, frame):
    print("\nAborted by signal!")
    abort_flag.set()


def listen_for_abort():
    """Listen for 'q' key to abort processing."""
    try:
        while not abort_flag.is_set():
            if sys.stdin.read(1).lower() == 'q':
                print("\nAbort signal received (q pressed)")
                abort_flag.set()
                break
    except Exception:
        pass


def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        orientation_value = exif.get(orientation)
        if orientation_value == 3:
            image = image.rotate(180, expand=True)
        elif orientation_value == 6:
            image = image.rotate(270, expand=True)
        elif orientation_value == 8:
            image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image


def get_clip_features(image_path):
    try:
        with Image.open(image_path) as img:
            img = correct_image_orientation(img)
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(img)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def collect_image_paths(folder_path, existing_images):
    image_paths = []
    folder_path = os.path.abspath(folder_path)

    for dirpath, _, filenames in os.walk(folder_path):
        if abort_flag.is_set():
            break
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                relative_path = os.path.relpath(full_path, folder_path)
                relative_path = relative_path.replace("\\", "/")
                if relative_path not in existing_images:
                    image_paths.append(relative_path)
    return image_paths


def process_images(folder_path, image_paths, feature_queue, process_progress):
    for image_path in image_paths:
        if abort_flag.is_set():
            break
        features = get_clip_features(os.path.join(folder_path, image_path))
        if features is not None:
            feature_queue.put((image_path, features))
            process_progress.update(1)
    feature_queue.put(None)


def write_to_database(feature_queue, db_filename, write_progress):
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (filename TEXT PRIMARY KEY, features BLOB)''')
    conn.commit()

    while True:
        item = feature_queue.get()
        if item is None:
            break
        image_path, features = item
        try:
            c.execute("INSERT OR REPLACE INTO images VALUES (?, ?)",
                      (image_path, features.tobytes()))
            write_progress.update(1)
        except sqlite3.Error as e:
            print(f"Database error for {image_path}: {e}")
        feature_queue.task_done()

    conn.commit()
    conn.close()


def create_database(folder_path):
    db_filename = 'tmp/image_features.db'

    # Get already processed images
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (filename TEXT PRIMARY KEY, features BLOB)''')
    c.execute('SELECT filename FROM images')
    existing_images = {row[0] for row in c.fetchall()}
    conn.close()

    print(f"Found {len(existing_images)} already processed images")

    image_paths = collect_image_paths(folder_path, existing_images)
    if not image_paths:
        print("No new images to process")
        return

    print(f"Found {len(image_paths)} new images")
    print("Press 'q' to abort...")

    process_progress = tqdm(total=len(image_paths), desc="Processing")
    write_progress = tqdm(total=len(image_paths), desc="Writing")

    signal.signal(signal.SIGINT, signal_handler)
    threading.Thread(target=listen_for_abort, daemon=True).start()

    t1 = threading.Thread(target=process_images, args=(folder_path, image_paths, feature_queue, process_progress))
    t2 = threading.Thread(target=write_to_database, args=(feature_queue, db_filename, write_progress))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    if abort_flag.is_set():
        print("\nProcessing aborted.")
    else:
        print("\nProcessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create CLIP image feature database.")
    parser.add_argument('--photos', required=True, type=str, help='Directory containing images')
    args = parser.parse_args()

    folder_path = os.path.abspath(args.photos)
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)

    create_database(folder_path)
