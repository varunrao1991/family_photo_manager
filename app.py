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
import argparse
from dotenv import load_dotenv
import logging
import struct

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration from environment
MAX_PAGE_SIZE = int(os.getenv('MAX_PAGE_SIZE', 50))
DATABASE = os.getenv('DATABASE', 'tmp/image_features.db')
CACHE_DIR = os.getenv('CACHE_DIR', 'tmp/cache/')
CLIP_MODEL = os.getenv('CLIP_MODEL', 'ViT-B/32')
DEVICE = os.getenv('DEVICE', 'auto')

# Determine device
if DEVICE == 'auto':
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = DEVICE

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize CLIP model
model, preprocess = clip.load(CLIP_MODEL, device=device)
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

@app.route('/')
def index():
    return render_template('index.html')

def compute_similarity(query_text=None, query_image=None):
    """Compute similarity between query and database images with proper feature normalization
    
    Args:
        query_text: Optional text query string
        query_image: Optional PIL Image object
    
    Returns:
        List of (filename, similarity_score) tuples sorted by descending similarity
    """
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    def normalize_features(features):
        """Normalize features to unit length for cosine similarity"""
        return features / features.norm(dim=-1, keepdim=True)

    try:
        with torch.no_grad():
            text_features = None
            image_features = None
            combined_features = None
            
            # Process text query if provided
            if query_text:
                text_inputs = clip.tokenize([query_text]).to(device)
                text_features = model.encode_text(text_inputs)
                text_features = normalize_features(text_features)
            
            # Process image query if provided
            if query_image:
                image_input = transform(query_image).unsqueeze(0).to(device)
                image_features = model.encode_image(image_input)
                image_features = normalize_features(image_features)
            
            # Validate at least one query type was provided
            if text_features is None and image_features is None:
                return []
            
            # Combine features for hybrid search
            if text_features is not None and image_features is not None:
                # Dynamic weighting based on feature magnitudes
                text_weight = 0.5 * (1 + text_features.norm().item())
                image_weight = 0.5 * (1 + image_features.norm().item())
                combined_features = (text_weight * text_features + image_weight * image_features) / 2
            elif text_features is not None:
                combined_features = text_features
            else:
                combined_features = image_features
            
            # Normalize final combined features
            combined_features = normalize_features(combined_features)
            
            # Get all database features at once for efficiency
            c.execute("SELECT filename, features FROM images")
            rows = c.fetchall()
            
            # Pre-allocate results list
            similarities = []
            
            # Process database features in batches if needed for large databases
            for row in rows:
                db_features = torch.frombuffer(row['features'], 
                                             dtype=torch.float32).to(device)
                db_features = normalize_features(db_features)
                
                # Cosine similarity
                similarity = torch.dot(combined_features.flatten(), db_features.flatten()).item()
                similarities.append((row['filename'], similarity))
            
            # Sort by descending similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
            
    except Exception as e:
        logging.error(f"Error computing similarities: {str(e)}")
        return []
    finally:
        conn.close()

@app.route('/similar-images', methods=['POST'])
def similar_images():
    query_text = request.form.get('query_text')
    query_image = request.files.get('query_image')
    page = int(request.form.get('page', 1))
    per_page = int(request.form.get('per_page', MAX_PAGE_SIZE))

    if not query_text and not query_image:
        return jsonify({'error': 'No query provided'}), 400

    # Process query image if provided
    img = None
    if query_image:
        img = Image.open(query_image).convert('RGB')

    similarities = compute_similarity(query_text, img)

    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paged_similarities = similarities[start_index:end_index]
    image_urls = [os.path.join(image[0]).replace('\\', '/') for image in paged_similarities]
    
    return jsonify({
        'similar_images': image_urls,
        'total_count': len(similarities),
        'search_type': 'hybrid' if query_text and query_image else 'text' if query_text else 'image'
    })

def if_exif_exists_reset(img):
    try:
        exif_data = img.info.get('exif')
        if exif_data:
            if not isinstance(exif_data, bytes) or len(exif_data) < 2:
                print("Warning: Invalid or too short EXIF data encountered.")
                return None  # Or handle differently, e.g., return original exif_data

            try:
                exif_dict = piexif.load(exif_data)
                exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
                return piexif.dump(exif_dict)
            except struct.error as e:
                print(f"Error loading EXIF data (struct error): {e}")
                return None
            except Exception as e:
                print(f"Error loading EXIF data (other): {e}")
                return None
        return None
    except (AttributeError, KeyError, IndexError) as e:
        print(f"Error accessing image info: {e}")
        return None

@app.route('/images/<path:filename>')
def serve_image(filename):
    image_path = os.path.join(app.config['IMAGE_DIR'], filename)
    
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

@app.route('/bigimages/<path:filename>')
def serve_image_big(filename):
    image_path = os.path.join(app.config['IMAGE_DIR'], filename)
    
    if os.path.isfile(image_path):
        return send_from_directory(app.config['IMAGE_DIR'], filename)
    else:
        abort(404)

def create_thumbnail(image_path, cache_path):
    with Image.open(image_path) as img:
        img.thumbnail((400, 400))
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
            file_path = os.path.join(app.config['IMAGE_DIR'], image_path)
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
            file_path = os.path.join(app.config['IMAGE_DIR'], image_path).replace("\\","/")
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
    parser = argparse.ArgumentParser(description='Image similarity search using CLIP')
    parser.add_argument('--photos', dest='image_dir', type=str, 
                       default=os.getenv('IMAGE_DIR'), 
                       help='Directory containing images (default: from .env)')
    parser.add_argument('--port', type=int, 
                       default=int(os.getenv('PORT', 5000)),
                       help='Port to run the server on (default: from .env)')
    args = parser.parse_args()

    app.config['IMAGE_DIR'] = os.path.abspath(args.image_dir)
    os.makedirs(app.config['IMAGE_DIR'], exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, port=args.port)