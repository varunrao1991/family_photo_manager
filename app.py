import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import logging
import threading
import queue
import sqlite3
import shutil
import time
import mimetypes
import subprocess
import hashlib
import json
from urllib.parse import quote
from collections import OrderedDict
from typing import Optional, List, Tuple, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, lru_cache
from dataclasses import dataclass
import traceback
import faiss

from flask import Flask, request, jsonify, send_from_directory, send_file, abort, render_template
import numpy as np
import torch
from PIL import Image, ImageOps, ExifTags
from pillow_heif import register_heif_opener
from clip import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import piexif
from dotenv import load_dotenv
import argparse

register_heif_opener()

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
    DISPLAY_CACHE_DIR: str = os.path.abspath(os.getenv('DISPLAY_CACHE_DIR', 'tmp/display-cache/'))
    BACKUP_DIR: str = os.path.abspath(os.getenv('BACKUP_DIR', 'tmp/backup/'))
    VIDEO_CONVERTED_DIR: str = os.path.abspath(os.getenv('VIDEO_CONVERTED_DIR', 'tmp/converted-videos'))
    VIDEO_POSTER_DIR: str = os.path.abspath(os.getenv('VIDEO_POSTER_DIR', 'tmp/video-posters'))
    CLIP_MODEL: str = os.getenv('CLIP_MODEL', 'ViT-B/32')
    DEVICE: str = os.getenv('DEVICE', 'auto')
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', 4))
    THUMBNAIL_SIZE: Tuple[int, int] = (400, 400)

    def __post_init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.DISPLAY_CACHE_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)
        os.makedirs(self.VIDEO_CONVERTED_DIR, exist_ok=True)
        os.makedirs(self.VIDEO_POSTER_DIR, exist_ok=True)

config = Config()


class SearchSuperseded(Exception):
    """Raised when a client has already started a newer search."""


@lru_cache(maxsize=8192)
def get_oriented_image_dimensions(image_path: str, modified_ns: int, file_size: int) -> Tuple[int, int]:
    """Read image dimensions once per file version without decoding pixels."""
    with Image.open(image_path) as source_image:
        width, height = source_image.size
        try:
            if source_image.getexif().get(274, 1) in (5, 6, 7, 8):
                width, height = height, width
        except Exception:
            pass
        return width, height

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
        self._similarity_cache = OrderedDict()
        self._similarity_cache_lock = threading.Lock()
        self._similarity_cache_ttl = 180
        self._similarity_cache_limit = 8
        # CLIP is expensive on the laptop CPU. Bound ranking work so request
        # bursts do not run several full matrix builds at the same time.
        self._similarity_compute_lock = threading.Semaphore(1)
        self._latest_search_lock = threading.Lock()
        self._latest_search_generation = {}
        self._clip_inference_lock = threading.Lock()
        self._video_conversion_lock = threading.Lock()
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
            c.execute('''CREATE TABLE IF NOT EXISTS videos (
                        filename TEXT PRIMARY KEY, file_size INTEGER NOT NULL, modified_ns INTEGER NOT NULL,
                        duration_seconds REAL NOT NULL, converted_filename TEXT,
                        frame_count INTEGER NOT NULL DEFAULT 0)''')
            c.execute('''CREATE TABLE IF NOT EXISTS video_embeddings (
                        video_filename TEXT NOT NULL, timestamp_seconds INTEGER NOT NULL,
                        features BLOB NOT NULL, poster_filename TEXT NOT NULL,
                        PRIMARY KEY (video_filename, timestamp_seconds))''')
        self._execute_db_operation(init_db)
        logger.info("Database initialized")

    def _setup_routes(self):
        """Configure all Flask routes."""
        self.app.route('/')(self.index)
        self.app.route('/similar-images', methods=['POST'])(self.similar_images)
        self.app.route('/search-cancel', methods=['POST'])(self.cancel_search)
        self.app.route('/find-duplicates', methods=['POST'])(self.find_duplicate_images)
        self.app.route('/images/<path:filename>')(self.serve_image)
        self.app.route('/bigimages/<path:filename>')(self.serve_image_big)
        self.app.route('/displayimages/<path:filename>')(self.serve_display_image)
        self.app.route('/videos/<path:filename>')(self.serve_video)
        self.app.route('/video-posters/<path:filename>')(self.serve_video_poster)
        self.app.route('/rotate-images', methods=['POST'])(self.rotate_images)
        self.app.route('/delete-images', methods=['DELETE'])(self.delete_images)

        @self.app.after_request
        def allow_tv_app(response):
            """Allow the bundled Tizen TV app to call this LAN-hosted API."""
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response

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
                with self._clip_inference_lock, torch.no_grad():
                    return self.model.encode_image(image).cpu().numpy().tobytes()
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def _ensure_thumbnail(self, image_path: str, filename: str) -> str:
        """Ensure a thumbnail exists for the given image."""
        cache_name = f"{filename}.jpg"
        cache_path = os.path.join(config.CACHE_DIR, cache_name)
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    rgba = img.convert('RGBA')
                    rgb = Image.new('RGB', rgba.size, '#000000')
                    rgb.paste(rgba, mask=rgba.getchannel('A'))
                    img = rgb
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                img.thumbnail(config.THUMBNAIL_SIZE)
                img.save(cache_path, 'JPEG', quality=85, optimize=True)
        return cache_path

    def _ensure_display_image(self, image_path: str, filename: str) -> str:
        """Create a TV-decodable, orientation-correct full-screen image once."""
        cache_name = f"{filename}.jpg"
        cache_path = os.path.join(config.DISPLAY_CACHE_DIR, cache_name)
        source_mtime = os.path.getmtime(image_path)
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= source_mtime:
            return cache_path

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image)
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, '#000000')
                background.paste(image, mask=image.getchannel('A'))
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            image.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
            image.save(cache_path, 'JPEG', quality=90, optimize=True)
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
        query_image_filename = request.form.get('query_image_filename', '').strip()
        weight = int(request.form.get('weight', 50)) / 100.0
        page = int(request.form.get('page', 1))
        per_page = min(int(request.form.get('per_page', config.MAX_PAGE_SIZE)), config.MAX_PAGE_SIZE)

        # A client reuses one generation for pagination. A newer generation
        # supersedes its older requests, including requests waiting for CLIP.
        client_id = request.form.get('client_id', '').strip()[:100]
        try:
            generation = int(request.form.get('search_generation', '0'))
        except (TypeError, ValueError):
            generation = 0
        if client_id and generation:
            with self._latest_search_lock:
                self._latest_search_generation[client_id] = max(
                    generation, self._latest_search_generation.get(client_id, 0)
                )

        def is_current_search():
            if not client_id or not generation:
                return True
            with self._latest_search_lock:
                return self._latest_search_generation.get(client_id, 0) <= generation

        if not is_current_search():
            return jsonify({'error': 'Search superseded'}), 409

        if not query_text and not query_image and not query_image_filename:
            return jsonify({'error': 'No query provided'}), 400

        # Process query image if provided
        img = None
        if query_image:
            try:
                with Image.open(query_image) as uploaded_image:
                    img = ImageOps.exif_transpose(uploaded_image).convert('RGB')
            except Exception as e:
                logger.error(f"Error processing query image: {str(e)}")
                return jsonify({'error': 'Invalid image provided'}), 400
        elif query_image_filename:
            try:
                image_root = os.path.abspath(self.app.config['IMAGE_DIR'])
                image_path = os.path.abspath(os.path.join(image_root, query_image_filename))
                if os.path.commonpath([image_root, image_path]) != image_root:
                    return jsonify({'error': 'Invalid image path'}), 400
                if not os.path.isfile(image_path):
                    return jsonify({'error': 'Image not found'}), 404

                # If query is a video file, use its sampled frame/poster from video_embeddings
                if query_image_filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')):
                    def get_video_poster(c):
                        c.execute("SELECT poster_filename FROM video_embeddings WHERE video_filename = ? LIMIT 1", (query_image_filename,))
                        row = c.fetchone()
                        return row['poster_filename'] if row else None
                    poster_name = self._execute_db_operation(get_video_poster)
                    if poster_name:
                        poster_path = os.path.join(config.VIDEO_POSTER_DIR, poster_name)
                        if os.path.isfile(poster_path):
                            with Image.open(poster_path) as source_image:
                                img = source_image.convert('RGB')
                    if img is None:
                        logger.warning(f"Could not find video poster for query video: {query_image_filename}")
                else:
                    display_path = self._ensure_display_image(
                        image_path, os.path.relpath(image_path, image_root)
                    )
                    with Image.open(display_path) as source_image:
                        img = source_image.convert('RGB')
            except ValueError:
                return jsonify({'error': 'Invalid image path'}), 400
            except Exception as e:
                logger.error(f"Error processing image search source: {str(e)}")
                return jsonify({'error': 'Could not read the selected image'}), 400

        # Compute similarities
        try:
            similarities = self._get_cached_similarities(query_text, img, weight, is_current_search)
        except SearchSuperseded:
            return jsonify({'error': 'Search superseded'}), 409

        if not is_current_search():
            return jsonify({'error': 'Search superseded'}), 409

        # Pagination and validation
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        paged_results = similarities[start_index:end_index]

        valid_images = []
        valid_results = []
        image_metadata = {}
        invalid_images = []

        for media_type, media_file, score, details in paged_results:
            if media_type == 'video':
                if not os.path.isfile(os.path.join(self.app.config['IMAGE_DIR'], media_file)):
                    continue
                valid_results.append({
                    'type': 'video', 'path': media_file,
                    'name': os.path.basename(media_file), 'score': score,
                    'duration_seconds': details['duration_seconds'],
                    'match_timestamp': details['timestamp_seconds'],
                    'poster_url': f"/video-posters/{details['poster_filename']}",
                    'video_url': '/videos/' + '/'.join(quote(part, safe='') for part in media_file.split('/'))
                })
                continue
            full_path = os.path.join(self.app.config['IMAGE_DIR'], media_file)
            if os.path.isfile(full_path):
                valid_images.append(media_file)
                valid_results.append({'type': 'image', 'path': media_file, 'score': score})
                try:
                    stat = os.stat(full_path)
                    width, height = get_oriented_image_dimensions(
                        full_path, stat.st_mtime_ns, stat.st_size
                    )
                    image_metadata[media_file] = {'width': width, 'height': height}
                except Exception as e:
                    logger.warning(f"Couldn't read dimensions for {media_file}: {str(e)}")
            else:
                logger.warning(f"Image file not found: {full_path}")
                invalid_images.append(media_file)

        # Clean up invalid images
        if invalid_images:
            self._delete_images(invalid_images, wait=False)

        return jsonify({
            'results': valid_results,
            'similar_images': valid_images,
            'image_metadata': image_metadata,
            'total_count': len(similarities),
            'page': page,
            'per_page': per_page,
            'search_type': 'hybrid' if query_text and query_image else 'text' if query_text else 'image'
        })

    def cancel_search(self):
        """Advance a client's search generation without starting new ranking work."""
        client_id = request.form.get('client_id', '').strip()[:100]
        try:
            generation = int(request.form.get('search_generation', '0'))
        except (TypeError, ValueError):
            generation = 0
        if not client_id or generation < 1:
            return jsonify({'error': 'Invalid cancellation request'}), 400
        with self._latest_search_lock:
            self._latest_search_generation[client_id] = max(
                generation, self._latest_search_generation.get(client_id, 0)
            )
        return jsonify({'cancelled': True})

    def find_duplicate_images(self):
        threshold = float(request.form.get('threshold', 0.95))
        page = int(request.form.get('page', 1))
        per_page = min(int(request.form.get('per_page', config.MAX_PAGE_SIZE)), config.MAX_PAGE_SIZE)
        
        result = self.group_duplicate_images(threshold, page, per_page)
        
        # Verify images exist (same as before)
        validated_groups = []
        missing_images = []
        
        for group in result['groups']:
            valid_group = []
            for image_file in group:
                full_path = os.path.join(self.app.config['IMAGE_DIR'], image_file)
                if os.path.isfile(full_path):
                    valid_group.append(image_file)
                else:
                    missing_images.append(image_file)
            if valid_group:
                validated_groups.append(valid_group)
        
        if missing_images:
            self._delete_images(missing_images, wait=False)
        
        return jsonify({
            'duplicate_groups': validated_groups,
            'total_groups': result['total_groups'],
            'total_images': result['total_images'],
            'page': result['page'],
            'per_page': result['per_page'],
            'threshold': threshold,
            'missing_images': len(missing_images)
        })

    def serve_image(self, filename: str):
        """Serve cached thumbnail images."""
        try:
            image_path = os.path.join(self.app.config['IMAGE_DIR'], filename)
            if not os.path.isfile(image_path):
                logger.warning(f"Image not found: {image_path}")
                abort(404)

            cache_path = self._ensure_thumbnail(image_path, filename)
            cache_name = os.path.relpath(cache_path, config.CACHE_DIR).replace('\\', '/')
            return send_from_directory(config.CACHE_DIR, cache_name, mimetype='image/jpeg')
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

    def serve_display_image(self, filename: str):
        """Serve a cached 1080p JPEG for smooth full-screen TV viewing."""
        try:
            image_path = os.path.join(self.app.config['IMAGE_DIR'], filename)
            if not os.path.isfile(image_path):
                logger.warning(f"Image not found: {image_path}")
                abort(404)
            self._ensure_display_image(image_path, filename)
            return send_from_directory(config.DISPLAY_CACHE_DIR, f"{filename}.jpg", mimetype='image/jpeg')
        except Exception as e:
            logger.error(f"Error serving display image {filename}: {str(e)}")
            abort(500)

    def serve_video(self, filename: str):
        """Serve an indexed original or its cached TV-compatible MP4 copy."""
        try:
            def lookup(c):
                c.execute('''SELECT filename, converted_filename, file_size, modified_ns
                             FROM videos WHERE filename = ?''', (filename,))
                return c.fetchone()
            row = self._execute_db_operation(lookup)
            if not row:
                abort(404)
            root = os.path.abspath(self.app.config['IMAGE_DIR'])
            source_path = os.path.abspath(os.path.join(root, row['filename']))
            if os.path.commonpath([root, source_path]) != root or not os.path.isfile(source_path):
                abort(404)
            source_stat = os.stat(source_path)
            source_changed = (row['file_size'], row['modified_ns']) != (source_stat.st_size, source_stat.st_mtime_ns)

            video_path = None
            if row['converted_filename'] and not source_changed:
                cached_path = os.path.join(config.VIDEO_CONVERTED_DIR, row['converted_filename'])
                if os.path.isfile(cached_path):
                    video_path = cached_path

            if video_path is None:
                with self._video_conversion_lock:
                    # Another request may have finished the conversion while we waited.
                    latest = self._execute_db_operation(lookup)
                    if latest and latest['converted_filename'] and not source_changed:
                        cached_path = os.path.join(config.VIDEO_CONVERTED_DIR, latest['converted_filename'])
                        if os.path.isfile(cached_path):
                            video_path = cached_path
                    if video_path is None:
                        video_path = self._convert_video_for_playback(source_path, filename)
                        if video_path != source_path:
                            converted_name = os.path.basename(video_path)
                            self._execute_db_operation(lambda c: c.execute(
                                'UPDATE videos SET converted_filename = ? WHERE filename = ?',
                                (converted_name, filename)
                            ))
                        elif source_changed:
                            self._execute_db_operation(lambda c: c.execute(
                                'UPDATE videos SET converted_filename = NULL WHERE filename = ?', (filename,)
                            ))

            content_type = 'video/mp4' if video_path != source_path else (
                mimetypes.guess_type(video_path)[0] or 'application/octet-stream'
            )
            if not os.path.isfile(video_path):
                abort(404)
            return send_file(video_path, mimetype=content_type, conditional=True, as_attachment=False)
        except ValueError:
            abort(404)
        except Exception as e:
            logger.error(f"Error serving video {filename}: {e}")
            abort(500)

    def _convert_video_for_playback(self, source_path: str, relative_filename: str) -> str:
        ffmpeg = shutil.which('ffmpeg')
        ffprobe = shutil.which('ffprobe')
        if not ffmpeg or not ffprobe:
            raise RuntimeError('FFmpeg/ffprobe are required to convert this video for TV playback')
        probe = subprocess.run(
            [ffprobe, '-v', 'error', '-show_entries',
             'format=format_name:stream=codec_type,codec_name,width,height,pix_fmt',
             '-of', 'json', source_path], capture_output=True, text=True, check=True
        )
        info = json.loads(probe.stdout)
        streams = info.get('streams', [])
        video = next((stream for stream in streams if stream.get('codec_type') == 'video'), None)
        if not video:
            raise ValueError('No video stream found')
        audio = next((stream for stream in streams if stream.get('codec_type') == 'audio'), None)
        formats = set((info.get('format', {}).get('format_name') or '').split(','))
        compatible = (
            os.path.splitext(source_path)[1].lower() == '.mp4'
            and 'mp4' in formats
            and video.get('codec_name') == 'h264'
            and video.get('pix_fmt') in ('yuv420p', 'yuvj420p')
            and int(video.get('width') or 0) <= 1920
            and int(video.get('height') or 0) <= 1080
            and (audio is None or audio.get('codec_name') == 'aac')
        )
        if compatible:
            return source_path

        os.makedirs(config.VIDEO_CONVERTED_DIR, exist_ok=True)
        stat = os.stat(source_path)
        cache_identity = f'{relative_filename}\0{stat.st_size}\0{stat.st_mtime_ns}'
        digest = hashlib.sha1(cache_identity.encode('utf-8')).hexdigest()[:20]
        converted_name = f'{digest}.mp4'
        converted_path = os.path.join(config.VIDEO_CONVERTED_DIR, converted_name)
        if os.path.isfile(converted_path):
            return converted_path
        partial_path = converted_path + '.partial.mp4'
        try:
            subprocess.run(
                [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-i', source_path,
                 '-map', '0:v:0', '-map', '0:a:0?', '-vf',
                 'scale=min(1920\\,iw):min(1080\\,ih):force_original_aspect_ratio=decrease:force_divisible_by=2',
                 '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                 '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', partial_path],
                capture_output=True, text=True, check=True
            )
            os.replace(partial_path, converted_path)
            return converted_path
        finally:
            if os.path.exists(partial_path):
                os.remove(partial_path)

    def serve_video_poster(self, filename: str):
        try:
            return send_from_directory(config.VIDEO_POSTER_DIR, filename, mimetype='image/jpeg')
        except Exception:
            abort(404)

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
                              query_image: Optional[Image.Image] = None,
                              weight: Optional[float] = 0.5,
                              is_current: Optional[Callable[[], bool]] = None) -> List[Tuple[str, str, float, Any]]:
        """Rank image vectors and each video's best sampled-frame vector."""
        def ensure_current():
            if is_current and not is_current():
                raise SearchSuperseded()

        def get_db_features(c):
            c.execute("SELECT filename, features FROM images")
            images = c.fetchall()
            c.execute('''SELECT v.filename, v.duration_seconds, e.timestamp_seconds,
                                e.features, e.poster_filename
                         FROM videos v JOIN video_embeddings e ON e.video_filename = v.filename''')
            return images, c.fetchall()

        try:
            ensure_current()
            text_features = None
            if query_text:
                text_inputs = clip.tokenize([query_text]).to(self.device)
                with self._clip_inference_lock, torch.no_grad():
                    encoded = self.model.encode_text(text_inputs)
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                    text_features = encoded.squeeze(0).cpu().numpy()
                ensure_current()

            image_features = None
            if query_image:
                image_input = self.transform(query_image).unsqueeze(0).to(self.device)
                with self._clip_inference_lock, torch.no_grad():
                    encoded = self.model.encode_image(image_input)
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                    image_features = encoded.squeeze(0).cpu().numpy()
                ensure_current()

            if text_features is None and image_features is None:
                return []

            image_rows, video_rows = self._execute_db_operation(get_db_features)
            ensure_current()
            expected_size = (text_features if text_features is not None else image_features).size
            results = []

            def score_vector_rows(rows, get_key, media_type):
                if not rows:
                    return
                row_vectors = []
                row_keys = []
                row_details = []
                for row in rows:
                    try:
                        vector = np.frombuffer(row['features'], dtype=np.float32)
                        if vector.size != expected_size:
                            continue
                        norm = np.linalg.norm(vector)
                        if not norm:
                            continue
                        row_vectors.append(vector / norm)
                        row_keys.append(get_key(row))
                        if media_type == 'video':
                            row_details.append({
                                'duration_seconds': row['duration_seconds'],
                                'timestamp_seconds': row['timestamp_seconds'],
                                'poster_filename': row['poster_filename'],
                            })
                        else:
                            row_details.append(None)
                    except Exception as e:
                        logger.warning('Skipping invalid %s embedding: %s', media_type, e)
                if not row_vectors:
                    return
                matrix = np.vstack(row_vectors)
                text_scores = matrix @ text_features if text_features is not None else None
                image_scores = matrix @ image_features if image_features is not None else None
                scores = ((1.0 - weight) * text_scores + weight * image_scores
                          if text_scores is not None and image_scores is not None
                          else text_scores if text_scores is not None else image_scores)
                if media_type == 'image':
                    results.extend(('image', row_keys[i], float(scores[i]), None)
                                   for i in np.argsort(scores)[::-1])
                else:
                    # A video appears once, with its best matching sampled frame.
                    best_per_video = {}
                    for i, key in enumerate(row_keys):
                        if key not in best_per_video or scores[i] > best_per_video[key][0]:
                            best_per_video[key] = (float(scores[i]), row_details[i])
                    results.extend(('video', key, score, details)
                                   for key, (score, details) in best_per_video.items())

            score_vector_rows(image_rows, lambda row: row['filename'], 'image')
            ensure_current()
            score_vector_rows(video_rows, lambda row: row['filename'], 'video')
            ensure_current()
            results.sort(key=lambda result: result[2], reverse=True)
            return results
        except SearchSuperseded:
            raise
        except Exception as e:
            logger.error(f"Error computing similarities: {str(e)}")
            logger.error("Stack trace:\n%s", traceback.format_exc())
            return []

    def _get_cached_similarities(self, query_text: Optional[str],
                                 query_image: Optional[Image.Image],
                                 weight: float,
                                 is_current: Optional[Callable[[], bool]] = None) -> List[Tuple[str, str, float, Any]]:
        """Reuse text-search rankings while clients request later pages."""
        cache_key = (query_text.strip(), round(weight, 4)) if query_text and query_image is None else None
        now = time.monotonic()
        def cached_result():
            if cache_key is None:
                return None
            with self._similarity_cache_lock:
                cached = self._similarity_cache.get(cache_key)
                if cached and time.monotonic() - cached[0] < self._similarity_cache_ttl:
                    self._similarity_cache.move_to_end(cache_key)
                    return cached[1]
                if cached:
                    del self._similarity_cache[cache_key]
            return None

        cached = cached_result()
        if cached is not None:
            return cached

        # Poll while waiting so a superseded request exits without spending a
        # full ranking slot. Recheck cache after acquiring: another identical
        # request may have completed while this one waited.
        while not self._similarity_compute_lock.acquire(timeout=0.1):
            if is_current and not is_current():
                raise SearchSuperseded()
        try:
            if is_current and not is_current():
                raise SearchSuperseded()
            cached = cached_result()
            if cached is not None:
                return cached

            similarities = self._compute_similarities(query_text, query_image, weight, is_current)
            if is_current and not is_current():
                raise SearchSuperseded()
            if cache_key is not None:
                with self._similarity_cache_lock:
                    self._similarity_cache[cache_key] = (time.monotonic(), similarities)
                    self._similarity_cache.move_to_end(cache_key)
                    while len(self._similarity_cache) > self._similarity_cache_limit:
                        self._similarity_cache.popitem(last=False)
            return similarities
        finally:
            self._similarity_compute_lock.release()

    def _invalidate_similarity_cache(self):
        with self._similarity_cache_lock:
            self._similarity_cache.clear()

    def group_duplicate_images(self, threshold: float = 0.95, page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """Group duplicate images using FAISS for efficient similarity search.
        
        Args:
            threshold: Minimum similarity score (0-1) to consider duplicates
            page: Page number for pagination
            per_page: Number of groups per page
            
        Returns:
            Dictionary containing:
            - groups: Paginated list of duplicate groups
            - total_groups: Total number of groups found
            - total_images: Total number of images in all groups
        """
        def get_all_image_features(c):
            c.execute("SELECT filename, features FROM images")
            return c.fetchall()

        try:
            # Get all image features from database
            rows = self._execute_db_operation(get_all_image_features)
            if not rows:
                return {'groups': [], 'total_groups': 0, 'total_images': 0}

            # Convert features to numpy array and normalize
            features = []
            valid_images = []
            for row in rows:
                try:
                    feature = torch.frombuffer(row['features'], dtype=torch.float32).numpy()
                    feature = feature / np.linalg.norm(feature)
                    features.append(feature)
                    valid_images.append(row['filename'])
                except Exception as e:
                    logger.error(f"Error processing image {row['filename']}: {str(e)}")
                    continue

            if not features:
                return {'groups': [], 'total_groups': 0, 'total_images': 0}

            features = np.array(features).astype('float32')
            
            # Build FAISS index
            dim = features.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
            index.add(features)

            # Search parameters
            k = min(100, len(features))  # Maximum possible neighbors to check
            visited = set()
            groups = []

            # Find duplicates for each image
            for i in range(len(features)):
                if i in visited:
                    continue

                # Find nearest neighbors
                distances, indices = index.search(features[i:i+1], k)
                similar_indices = [idx for idx, dist in zip(indices[0], distances[0]) 
                                if idx != i and dist >= threshold and idx not in visited]

                if similar_indices:
                    group = {i} | set(similar_indices)
                    visited.update(group)
                    groups.append([valid_images[idx] for idx in sorted(group)])

            # Sort groups by size (descending) and then by first filename
            groups.sort(key=lambda x: (-len(x), x[0]))

            # Pagination
            total_groups = len(groups)
            total_images = sum(len(g) for g in groups)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_groups = groups[start_idx:end_idx]

            return {
                'groups': paginated_groups,
                'total_groups': total_groups,
                'total_images': total_images,
                'page': page,
                'per_page': per_page
            }

        except Exception as e:
            logger.error(f"Error in FAISS duplicate grouping: {str(e)}")
            logger.error("Stack trace:\n%s", traceback.format_exc())
            return {'groups': [], 'total_groups': 0, 'total_images': 0}

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
        self._invalidate_similarity_cache()
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
        self._invalidate_similarity_cache()

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
                       default=os.getenv('IMAGE_DIR', 'D:/images'), 
                       help='Directory containing images (default: from .env or D:/images)')
    parser.add_argument('--port', type=int, 
                       default=int(os.getenv('PORT', 5000)),
                       help='Port to run the server on (default: from .env or 5000)')
    args = parser.parse_args()

    image_dir = args.image_dir or os.getenv('IMAGE_DIR', 'D:/images')

    service = ImageSearchService()
    service.app.config['IMAGE_DIR'] = os.path.abspath(image_dir)
    os.makedirs(service.app.config['IMAGE_DIR'], exist_ok=True)

    try:
        logger.info(f"Starting server on port {args.port} with image directory {service.app.config['IMAGE_DIR']}")
        service.app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        service.shutdown()
