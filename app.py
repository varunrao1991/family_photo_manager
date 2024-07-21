from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import sqlite3
import numpy as np
import torch
from PIL import Image, ExifTags
from clip import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os
import threading
import queue
import piexif

app = Flask(__name__)

MAX_PAGE_SIZE = 50
IMAGE_DIR = 'F:/'
DATABASE = 'image_similarity.db'
CACHE_DIR = 'cache/'

# Ensure the cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

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

# Queue and lock for database operations
db_queue = queue.Queue()
db_lock = threading.Lock()

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

# Function to process database operations
def process_db_operations():
    while True:
        operation = db_queue.get()
        if operation is None:  # Sentinel value to exit
            break
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        operation(c)
        conn.commit()
        conn.close()
        db_queue.task_done()

# Start the background thread for database operations
db_thread = threading.Thread(target=process_db_operations, daemon=True)
db_thread.start()

def queue_db_operation(operation):
    with db_lock:
        db_queue.put(operation)

def compute_similarity(query_text, query_image=None):
    similarities = []
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if query_image:
        image = transform(query_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            c.execute("SELECT * FROM images")
            rows = c.fetchall()
            for row in rows:
                db_features = np.frombuffer(row['features'], dtype=np.float32)
                similarity = (image_features @ torch.tensor(db_features).to(device)).cpu().numpy().item()
                similarities.append((row['filename'], similarity))
    else:
        with torch.no_grad():
            text = clip.tokenize([query_text]).to(device)
            text_features = model.encode_text(text)
            c.execute("SELECT * FROM images")
            rows = c.fetchall()
            for row in rows:
                db_features = np.frombuffer(row['features'], dtype=np.float32)
                similarity = (text_features @ torch.tensor(db_features).to(device)).cpu().numpy().item()
                similarities.append((row['filename'], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    conn.close()
    return similarities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/similar-images', methods=['POST'])
def similar_images():
    query_text = request.form.get('query_text')
    query_image = request.files.get('query_image')
    page = int(request.form.get('page', 1))
    per_page = int(request.form.get('per_page', MAX_PAGE_SIZE))

    if query_text:
        similarities = compute_similarity(query_text)
    elif query_image:
        query_image = Image.open(query_image).convert('RGB')
        similarities = compute_similarity(None, query_image)

    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paged_similarities = similarities[start_index:end_index]
    image_urls = [os.path.relpath(image[0], IMAGE_DIR).replace('\\', '/') for image in paged_similarities]
    
    return jsonify({'similar_images': image_urls, 'total_count': len(similarities)})

def if_exif_exists_reset(img):
    try:
        exif_data = img.info.get('exif', None)
        if exif_data:
            exif_data = piexif.load(exif_data)
            
            exif_data['0th'][piexif.ImageIFD.Orientation] = 1
            
            return piexif.dump(exif_data)
    except (AttributeError, KeyError, IndexError) as e:
        print(f"Error: {e}")
    return None

@app.route('/images/<path:filename>')
def serve_image(filename):
    image_path = os.path.join(IMAGE_DIR, filename)
    
    if os.path.isfile(image_path):
        with Image.open(image_path) as img:
            exif_bytes = if_exif_exists_reset(img)
            if exif_bytes is None:
                img.save(image_path)
            else:
                img = correct_image_orientation(img)
                img.save(image_path, exif=exif_bytes)          
        cache_path = os.path.join(CACHE_DIR, filename)
        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if not os.path.isfile(cache_path):
            create_thumbnail(image_path, cache_path)
        return send_from_directory(CACHE_DIR, filename)
    else:
        abort(404)

def create_thumbnail(image_path, cache_path):
    with Image.open(image_path) as img:
        img.thumbnail((300, 300))
        img.save(cache_path)

@app.route('/rotate-images', methods=['POST'])
def rotate_images():
    data = request.get_json()
    image_paths = data.get('images', [])
    direction = data.get('direction', 'left')
    
    angles = {'left': 90, 'right': -90}
    angle = angles[direction]

    if not image_paths or direction not in ['left', 'right']:
        return jsonify({'success': False, 'message': 'Invalid request'}), 400

    def rotate_operations(c):
        for image_path in image_paths:
            file_path = os.path.join(IMAGE_DIR, image_path)
            cache_path = os.path.join(CACHE_DIR, image_path)
            
            # Rotate the image
            with Image.open(file_path) as img:
                img = img.rotate(angle, expand=True)
                exif_bytes = if_exif_exists_reset(img)
                if exif_bytes is None:
                    img.save(file_path)
                else:
                    img.save(file_path, exif=exif_bytes)              

            create_thumbnail(file_path, cache_path)

            # Recalculate features
            with Image.open(file_path) as img:
                image = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image).cpu().numpy().tobytes()

            c.execute("UPDATE images SET features = ? WHERE filename = ?", (image_features, file_path))
    
    queue_db_operation(rotate_operations)
    return jsonify({'success': True, 'message': f'Selected images rotated {direction}'})

@app.route('/delete-images', methods=['DELETE'])
def delete_images():
    data = request.get_json()
    image_paths = data.get('images', [])

    def delete_operation(c):
        for image_path in image_paths:
            file_path = os.path.join(IMAGE_DIR, image_path)
            print(file_path)
            c.execute("DELETE FROM images WHERE filename = ?", (file_path,))
            if os.path.isfile(file_path):
                os.remove(file_path)
            cache_file_path = os.path.join(CACHE_DIR, image_path)
            if os.path.isfile(cache_file_path):
                os.remove(cache_file_path)

    queue_db_operation(delete_operation)
    return jsonify({'message': 'Selected images have been deleted.'})

if __name__ == '__main__':
    app.run()