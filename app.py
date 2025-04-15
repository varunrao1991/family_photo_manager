import os
import logging
import threading
import queue
import sqlite3
import shutil
from typing import Optional, List, Tuple, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from dataclasses import dataclass
import traceback

from flask import Flask, request, jsonify, send_from_directory, abort, render_template
import numpy as np
import torch
from PIL import Image, ExifTags
from clip import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import piexif
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# ======================
# Configuration
# ======================
@dataclass
class Config:
    MAX_PAGE_SIZE: int = int(os.getenv('MAX_PAGE_SIZE', 50))
    DATABASE: str = os.getenv('DATABASE', 'tmp/image_features.db')
    CACHE_DIR: str = os.path.abspath(os.getenv('CACHE_DIR', 'tmp/cache/'))
    BACKUP_DIR: str = os.path.abspath(os.getenv('BACKUP_DIR', 'tmp/backup/'))
    CLIP_MODEL: str = os.getenv('CLIP_MODEL', 'ViT-B/32')
    DEVICE: str = os.getenv('DEVICE', 'auto')
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', 4))
    THUMBNAIL_SIZE: Tuple[int, int] = (400, 400)

    def __post_init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)

config = Config()

# ======================
# Logging Setup
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================
# Application Core
# ======================
class ImageSearchService:
    def __init__(self):
        # Thread pool for synchronous operations
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
        # Operation queue and processing thread
        self.op_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.op_thread = threading.Thread(target=self._process_operations, daemon=True)
        self.op_thread.start()
        
        self.app = Flask(__name__)
        self._init_clip_model()
        self._init_database()
        self._setup_routes()

    def _init_clip_model(self):
        """Initialize the CLIP model for image/text embeddings."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu" if config.DEVICE == 'auto' else config.DEVICE
            self.model, self.preprocess = clip.load(config.CLIP_MODEL, device=device)
            self.device = device
            self.model.eval()
            self.transform = Compose([
                Resize(224),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            logger.info(f"Loaded CLIP model {config.CLIP_MODEL} on device {device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise

    def _init_database(self):
        """Initialize the database with required tables."""
        def init_db(c):
            c.execute('''CREATE TABLE IF NOT EXISTS images
                        (filename TEXT PRIMARY KEY,
                         features BLOB,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        self._execute_db_operation(init_db)
        logger.info("Database initialized")

    def _setup_routes(self):
        """Configure all Flask routes."""
        self.app.route('/')(self.index)
        self.app.route('/similar-images', methods=['POST'])(self.similar_images)
        self.app.route('/images/<path:filename>')(self.serve_image)
        self.app.route('/bigimages/<path:filename>')(self.serve_image_big)
        self.app.route('/rotate-images', methods=['POST'])(self.rotate_images)
        self.app.route('/delete-images', methods=['DELETE'])(self.delete_images)

    # ======================
    # Database Operations
    # ======================
    def _execute_db_operation(self, operation: Callable, wait: bool = True):
        """Execute a database operation either synchronously or asynchronously."""
        if wait:
            # For synchronous operations, bypass the executor and run directly
            conn = None
            try:
                conn = sqlite3.connect(config.DATABASE)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                result = operation(c)
                conn.commit()
                return result
            except sqlite3.Error as e:
                logger.error(f"Database operation failed: {str(e)}")
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()
        else:
            # For async operations, use the queue
            self.op_queue.put(('db', operation))

    def _process_operations(self):
        """Background thread for processing queued operations."""
        while not self.stop_event.is_set():
            try:
                op_type, operation = self.op_queue.get(timeout=1)
                if op_type == 'db':
                    self._run_db_operation(operation)
                self.op_queue.task_done()
            except queue.Empty:
                continue

    def _run_db_operation(self, operation: Callable):
        """Execute a database operation with proper connection handling."""
        conn = None
        try:
            conn = sqlite3.connect(config.DATABASE)
            c = conn.cursor()
            operation(c)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database operation failed: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    # ======================
    # Image Processing
    # ======================
    @staticmethod
    def correct_image_orientation(image: Image.Image) -> Image.Image:
        """Correct image orientation based on EXIF data."""
        try:
            exif = image._getexif()
            if not exif:
                return image

            orientation_key = next(
                (key for key, value in ExifTags.TAGS.items() if value == 'Orientation'), None
            )

            if orientation_key is None or orientation_key not in exif:
                return image

            orientation = exif[orientation_key]

            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

        except Exception:
            pass
        return image

    def _process_image_file(self, image_path: str) -> Optional[bytes]:
        """Process an image file and return its feature vector."""
        try:
            with Image.open(image_path) as img:
                img = self.correct_image_orientation(img)
                image = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    return self.model.encode_image(image).cpu().numpy().tobytes()
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def _ensure_thumbnail(self, image_path: str, filename: str) -> str:
        """Ensure a thumbnail exists for the given image."""
        cache_path = os.path.join(config.CACHE_DIR, filename)
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with Image.open(image_path) as img:
                img = self.correct_image_orientation(img)
                img.thumbnail(config.THUMBNAIL_SIZE)
                img.save(cache_path, quality=85)
        return cache_path

    # ======================
    # API Endpoints
    # ======================
    def index(self):
        return render_template('index.html')

    def similar_images(self):
        """Handle similarity search requests."""
        query_text = request.form.get('query_text')
        query_image = request.files.get('query_image')
        weight = int(request.form.get('weight', 50)) / 100.0
        page = int(request.form.get('page', 1))
        per_page = min(int(request.form.get('per_page', config.MAX_PAGE_SIZE)), config.MAX_PAGE_SIZE)

        if not query_text and not query_image:
            return jsonify({'error': 'No query provided'}), 400

        # Process query image if provided
        img = None
        if query_image:
            try:
                img = Image.open(query_image).convert('RGB')
                img = self.correct_image_orientation(img)
            except Exception as e:
                logger.error(f"Error processing query image: {str(e)}")
                return jsonify({'error': 'Invalid image provided'}), 400

        # Compute similarities
        similarities = self._compute_similarities(query_text, img, weight)

        # Pagination and validation
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        paged_results = similarities[start_index:end_index]

        valid_images = []
        invalid_images = []
        
        for image_file, _ in paged_results:
            full_path = os.path.join(self.app.config['IMAGE_DIR'], image_file)
            if os.path.isfile(full_path):
                valid_images.append(image_file)
            else:
                logger.warning(f"Image file not found: {full_path}")
                invalid_images.append(image_file)

        # Clean up invalid images
        if invalid_images:
            self._delete_images(invalid_images, wait=False)

        return jsonify({
            'similar_images': valid_images,
            'total_count': len(similarities),
            'page': page,
            'per_page': per_page,
            'search_type': 'hybrid' if query_text and query_image else 'text' if query_text else 'image'
        })

    def serve_image(self, filename: str):
        """Serve cached thumbnail images."""
        try:
            image_path = os.path.join(self.app.config['IMAGE_DIR'], filename)
            if not os.path.isfile(image_path):
                logger.warning(f"Image not found: {image_path}")
                abort(404)

            cache_path = self._ensure_thumbnail(image_path, filename)
            return send_from_directory(config.CACHE_DIR, filename)
        except Exception as e:
            logger.error(f"Error serving image {filename}: {str(e)}")
            abort(500)

    def serve_image_big(self, filename: str):
        """Serve original full-size images."""
        try:
            image_path = os.path.join(self.app.config['IMAGE_DIR'], filename)
            if not os.path.isfile(image_path):
                logger.warning(f"Image not found: {image_path}")
                abort(404)
            return send_from_directory(self.app.config['IMAGE_DIR'], filename)
        except Exception as e:
            logger.error(f"Error serving full image {filename}: {str(e)}")
            abort(500)

    def rotate_images(self):
        """Handle image rotation requests."""
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'Invalid request'}), 400

        image_paths = data.get('images', [])
        direction = data.get('direction', 'left')
        
        if not image_paths or direction not in ['left', 'right']:
            return jsonify({'success': False, 'message': 'Invalid request'}), 400

        # Execute synchronously
        result = self._rotate_images(image_paths, direction)
        return jsonify(result)

    def delete_images(self):
        """Handle image deletion requests."""
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'Invalid request'}), 400

        image_paths = data.get('images', [])
        if not image_paths:
            return jsonify({'success': False, 'message': 'No images specified'}), 400

        # Execute synchronously
        result = self._delete_images(image_paths, wait=True)
        return jsonify(result)

    # ======================
    # Core Operations
    # ======================
    def _compute_similarities(self, query_text: Optional[str] = None, 
                            query_image: Optional[Image.Image] = None, weight: Optional[float] = 0.5) -> List[Tuple[str, float]]:
        """Compute similarity scores between query and database images."""
        def get_db_features(c):
            c.execute("SELECT filename, features FROM images")
            data = c.fetchall()
            return data

        try:
            # Get text features if text query provided
            text_features = None
            if query_text:
                text_inputs = clip.tokenize([query_text]).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.squeeze(0)  # Remove batch dimension

            # Get image features if image query provided
            image_features = None
            if query_image:
                image_input = self.transform(query_image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    image_features = image_features.squeeze(0)  # Remove batch dimension

            if text_features is None and image_features is None:
                return []

            # Get all database features
            rows = self._execute_db_operation(get_db_features)
            similarities = []
            
            for row in rows:
                try:
                    db_features = torch.frombuffer(row['features'], dtype=torch.float32).to(self.device)
                    db_features = db_features / db_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarities
                    text_sim = 0.0
                    image_sim = 0.0
                    
                    if text_features is not None:
                        text_sim = torch.dot(text_features, db_features).item()
                    
                    if image_features is not None:
                        image_sim = torch.dot(image_features, db_features).item()
                    
                    # Combined score strategy
                    if text_features is not None and image_features is not None:
                        # Weighted combination (adjust weights as needed)
                        combined_score = (1.0- weight) * text_sim + weight * image_sim
                    elif text_features is not None:
                        combined_score = text_sim
                    else:
                        combined_score = image_sim
                    
                    similarities.append((row['filename'], combined_score))
                except Exception as e:
                    logger.error(f"Error processing image {row['filename']}: {str(e)}")
                    continue
            
            return sorted(similarities, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error computing similarities: {str(e)}")
            logger.error("Stack trace:\n%s", traceback.format_exc())
            return []

    def _rotate_images(self, image_paths: List[str], direction: str) -> Dict[str, Any]:
        """Rotate images and update their features in the database."""
        angle = 90 if direction == 'left' else -90
        success_count = 0

        def rotate_operation(c):
            nonlocal success_count
            for image_file in image_paths:
                try:
                    file_path = os.path.join(self.app.config['IMAGE_DIR'], image_file)
                    cache_path = os.path.join(config.CACHE_DIR, image_file)
                    
                    # Open and rotate image
                    with Image.open(file_path) as img:
                        img_format = img.format
                        img_mode = img.mode
                        img = self.correct_image_orientation(img)
                        
                        # Rotate and save with original settings
                        rotated_img = img.rotate(angle, expand=True)
                        save_kwargs = {
                            'quality': 95,
                            'format': img_format,
                            'subsampling': 0
                        }
                        
                        # Handle EXIF data
                        if 'exif' in img.info:
                            try:
                                exif_dict = piexif.load(img.info['exif'])
                                exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
                                save_kwargs['exif'] = piexif.dump(exif_dict)
                            except Exception as ex:
                                logger.warning(f"Couldn't process EXIF for {image_file}: {str(ex)}")
                                save_kwargs['exif'] = img.info['exif']
                        
                        if img_mode == 'RGB' and rotated_img.mode != 'RGB':
                            rotated_img = rotated_img.convert('RGB')
                        
                        rotated_img.save(file_path, **save_kwargs)

                    # Update cache
                    if os.path.exists(cache_path):
                        with Image.open(file_path) as img:
                            img.thumbnail(config.THUMBNAIL_SIZE)
                            img.save(cache_path, quality=85, format=img_format, subsampling=0)

                    # Update features in database
                    features = self._process_image_file(file_path)
                    if features:
                        c.execute("UPDATE images SET features = ? WHERE filename = ?", 
                                (features, image_file))
                        success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to rotate {image_file}: {str(e)}")

        self._execute_db_operation(rotate_operation)
        return {
            'success': True,
            'count': success_count,
            'total': len(image_paths),
            'message': f'Rotated {success_count}/{len(image_paths)} images'
        }

    def _delete_images(self, image_paths: List[str], wait: bool = True) -> Dict[str, Any]:
        """Delete images and their database entries."""
        success_count = 0

        def delete_operation(c):
            nonlocal success_count
            for image_file in image_paths:
                try:
                    file_path = os.path.join(self.app.config['IMAGE_DIR'], image_file)
                    cache_path = os.path.join(config.CACHE_DIR, image_file)
                    
                    # Backup before deletion
                    backup_path = os.path.join(config.BACKUP_DIR, image_file)
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    
                    if os.path.isfile(file_path):
                        shutil.copy2(file_path, backup_path)
                    
                    # Delete from database
                    c.execute("DELETE FROM images WHERE filename = ?", (image_file,))
                    
                    # Delete files
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to delete {image_file}: {str(e)}")

        if wait:
            self._execute_db_operation(delete_operation)
        else:
            self._execute_db_operation(delete_operation, wait=False)

        return {
            'success': True,
            'count': success_count,
            'total': len(image_paths),
            'message': f'Deleted {success_count}/{len(image_paths)} images'
        }

    def shutdown(self):
        """Clean shutdown of the service."""
        self.stop_event.set()
        self.op_queue.put(None)
        self.op_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Service shutdown complete")

# ======================
# Application Factory
# ======================
def create_app(image_dir: str) -> Flask:
    """Create and configure the Flask application."""
    service = ImageSearchService()
    service.app.config['IMAGE_DIR'] = os.path.abspath(image_dir)
    os.makedirs(service.app.config['IMAGE_DIR'], exist_ok=True)
    return service.app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image similarity search using CLIP')
    parser.add_argument('--photos', dest='image_dir', type=str, 
                       default=os.getenv('IMAGE_DIR'), 
                       help='Directory containing images (default: from .env)')
    parser.add_argument('--port', type=int, 
                       default=int(os.getenv('PORT', 5000)),
                       help='Port to run the server on (default: from .env)')
    args = parser.parse_args()

    if not args.image_dir:
        logger.error("No image directory specified. Use --photos or set IMAGE_DIR in .env")
        exit(1)

    service = ImageSearchService()
    service.app.config['IMAGE_DIR'] = os.path.abspath(args.image_dir)
    os.makedirs(service.app.config['IMAGE_DIR'], exist_ok=True)

    try:
        logger.info(f"Starting server on port {args.port} with image directory {service.app.config['IMAGE_DIR']}")
        service.app.run(host='0.0.0.0', port=args.port)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        service.shutdown()