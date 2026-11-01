import os
import sys
import signal
import argparse
import sqlite3
import threading
import subprocess
import json
import shutil
import hashlib
import tempfile
from queue import Queue
from tqdm import tqdm
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip

register_heif_opener()

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

# Videos that need transcoding are converted once, when their CLIP frames are
# indexed. Playback and poster assets live outside the original photo folder.
VIDEO_POSTER_DIR = os.path.abspath(os.getenv('VIDEO_POSTER_DIR', 'tmp/video-posters'))
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.m4v', '.avi', '.mkv', '.3gp', '.webm', '.wmv', '.mpeg', '.mpg', '.mts', '.m2ts', '.3g2')

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
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(img)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def get_clip_features_batch(image_paths):
    """Encode a video's sampled frames together to avoid per-frame CLIP overhead."""
    tensors = []
    valid_indices = []
    for index, image_path in enumerate(image_paths):
        try:
            with Image.open(image_path) as source:
                image = correct_image_orientation(source).convert('RGB')
                tensors.append(transform(image))
                valid_indices.append(index)
        except Exception as error:
            print(f'Error reading sampled video frame {image_path}: {error}')
    results = [None] * len(image_paths)
    if not tensors:
        return results
    try:
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(batch).cpu().numpy()
        for index, vector in zip(valid_indices, features):
            results[index] = vector.flatten()
    except Exception as error:
        print(f'Batch CLIP encoding failed; falling back to single-frame encoding: {error}')
        for index in valid_indices:
            results[index] = get_clip_features(image_paths[index])
    return results


def collect_image_paths(folder_path, existing_images):
    image_paths = []
    folder_path = os.path.abspath(folder_path)

    for dirpath, _, filenames in os.walk(folder_path):
        if abort_flag.is_set():
            break
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif')):
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                relative_path = os.path.relpath(full_path, folder_path)
                relative_path = relative_path.replace("\\", "/")
                if relative_path not in existing_images:
                    image_paths.append(relative_path)
    return image_paths


def _video_tools():
    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('FFmpeg and ffprobe are required for video indexing. Install FFmpeg and add it to PATH.')
    return ffmpeg, ffprobe


def _probe_video(path, ffprobe):
    completed = subprocess.run(
        [ffprobe, '-v', 'error', '-show_entries',
         'format=duration:stream=codec_type',
         '-of', 'json', path], capture_output=True, text=True, check=True
    )
    info = json.loads(completed.stdout)
    if not any(item.get('codec_type') == 'video' for item in info.get('streams', [])):
        raise ValueError('No video stream found')
    return float(info.get('format', {}).get('duration') or 0)


def _sample_video_timestamps(duration):
    duration = max(0.0, float(duration or 0))
    if duration < 300:
        return [0.0, duration / 2.0] if duration > 0 else [0.0]
    interval = duration / 5.0
    return [interval * index for index in range(5)]


def _extract_video_frames(source_path, poster_prefix, duration, ffmpeg):
    os.makedirs(VIDEO_POSTER_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='clip-video-frames-') as scratch:
        output = []
        timestamps = _sample_video_timestamps(duration)
        command = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error']
        frame_paths = []
        # One FFmpeg process per video; each input seeks directly to its sample
        # point, avoiding full-duration decode and repeated process startup.
        for index, timestamp in enumerate(timestamps):
            command.extend(['-ss', f'{timestamp:.3f}', '-i', source_path])
            frame_paths.append(os.path.join(scratch, f'frame-{index:02d}.jpg'))
        for index, frame_path in enumerate(frame_paths):
            command.extend(['-map', f'{index}:v:0', '-frames:v', '1', '-vf', 'scale=640:-2',
                            '-q:v', '3', frame_path])
        subprocess.run(command, capture_output=True, text=True)
        available = [(timestamp, frame_path) for timestamp, frame_path in zip(timestamps, frame_paths)
                     if os.path.isfile(frame_path)]
        for timestamp, frame_path in available:
            stamp = int(round(timestamp * 1000))
            poster_name = f'{poster_prefix}-{stamp:012d}.jpg'
            poster_path = os.path.join(VIDEO_POSTER_DIR, poster_name)
            shutil.copyfile(frame_path, poster_path)
        batch_features = get_clip_features_batch([frame_path for _, frame_path in available])
        for (timestamp, frame_path), features in zip(available, batch_features):
            stamp = int(round(timestamp * 1000))
            poster_name = f'{poster_prefix}-{stamp:012d}.jpg'
            if features is not None:
                output.append((timestamp, poster_name, features))
        if not output:
            raise ValueError('No frames could be extracted')
        return output


def collect_video_paths(folder_path):
    paths = []
    for dirpath, _, filenames in os.walk(os.path.abspath(folder_path)):
        if abort_flag.is_set():
            break
        for filename in filenames:
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                paths.append(os.path.relpath(full_path, folder_path).replace('\\', '/'))
    return paths


def process_videos(folder_path, db_filename):
    videos = collect_video_paths(folder_path)
    if not videos:
        print('No videos found to index')
        return
    try:
        ffmpeg, ffprobe = _video_tools()
    except RuntimeError as error:
        print(f'Skipping video indexing: {error}')
        return

    conn = sqlite3.connect(db_filename)
    conn.execute('''CREATE TABLE IF NOT EXISTS videos (
        filename TEXT PRIMARY KEY, file_size INTEGER NOT NULL, modified_ns INTEGER NOT NULL,
        duration_seconds REAL NOT NULL, converted_filename TEXT, frame_count INTEGER NOT NULL DEFAULT 0)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS video_embeddings (
        video_filename TEXT NOT NULL, timestamp_seconds INTEGER NOT NULL, features BLOB NOT NULL,
        poster_filename TEXT NOT NULL, PRIMARY KEY (video_filename, timestamp_seconds))''')
    conn.commit()
    print(f'Found {len(videos)} videos; sampling 2 frames under five minutes, otherwise 5')

    current_paths = set(videos)
    known_paths = {row[0] for row in conn.execute('SELECT filename FROM videos').fetchall()}
    for removed_path in known_paths - current_paths:
        old_posters = conn.execute(
            'SELECT poster_filename FROM video_embeddings WHERE video_filename = ?', (removed_path,)
        ).fetchall()
        for (poster_name,) in old_posters:
            try:
                os.remove(os.path.join(VIDEO_POSTER_DIR, poster_name))
            except FileNotFoundError:
                pass
        conn.execute('DELETE FROM video_embeddings WHERE video_filename = ?', (removed_path,))
        conn.execute('DELETE FROM videos WHERE filename = ?', (removed_path,))
    conn.commit()

    for relative_path in tqdm(videos, desc='Indexing videos'):
        if abort_flag.is_set():
            break
        source_path = os.path.join(folder_path, relative_path)
        try:
            stat = os.stat(source_path)
            existing = conn.execute(
                'SELECT file_size, modified_ns, duration_seconds FROM videos WHERE filename = ?', (relative_path,)
            ).fetchone()
            if existing and existing[:2] == (stat.st_size, stat.st_mtime_ns):
                stored_times = [round(row[0], 3) for row in conn.execute(
                    'SELECT timestamp_seconds FROM video_embeddings WHERE video_filename = ? ORDER BY timestamp_seconds',
                    (relative_path,)).fetchall()]
                expected_times = [round(value, 3) for value in _sample_video_timestamps(existing[2])]
                if stored_times == expected_times:
                    continue
            old_posters = conn.execute(
                'SELECT poster_filename FROM video_embeddings WHERE video_filename = ?', (relative_path,)
            ).fetchall()
            for (poster_name,) in old_posters:
                try:
                    os.remove(os.path.join(VIDEO_POSTER_DIR, poster_name))
                except FileNotFoundError:
                    pass
            conn.execute('DELETE FROM video_embeddings WHERE video_filename = ?', (relative_path,))
            digest = hashlib.sha1(relative_path.encode('utf-8')).hexdigest()[:20]
            duration = _probe_video(source_path, ffprobe)
            # Always embed frames from the original. A TV-compatible copy is
            # produced later, and only if the user actually starts playback.
            converted_name = None
            frame_rows = _extract_video_frames(source_path, digest, duration, ffmpeg)
            conn.execute('DELETE FROM videos WHERE filename = ?', (relative_path,))
            conn.execute('''INSERT INTO videos
                (filename, file_size, modified_ns, duration_seconds, converted_filename, frame_count)
                VALUES (?, ?, ?, ?, ?, ?)''',
                (relative_path, stat.st_size, stat.st_mtime_ns, duration, converted_name, len(frame_rows)))
            conn.executemany('''INSERT INTO video_embeddings
                (video_filename, timestamp_seconds, features, poster_filename) VALUES (?, ?, ?, ?)''',
                [(relative_path, timestamp, features.tobytes(), poster)
                 for timestamp, poster, features in frame_rows])
            conn.commit()
        except Exception as error:
            conn.rollback()
            print(f'Video indexing error for {relative_path}: {error}')
    conn.close()


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
            c.execute("INSERT OR REPLACE INTO images (filename, features) VALUES (?, ?)",
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
    else:
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

    if not abort_flag.is_set():
        process_videos(folder_path, db_filename)

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
