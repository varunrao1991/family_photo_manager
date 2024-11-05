import time 
import os
import signal
import argparse
import sqlite3
import tqdm
import face_recognition
from queue import Queue
import threading
import sys
import pickle  # To serialize face locations
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

progress_lock = threading.Lock()

# Global variables for synchronization and signal handling
face_feature_queue = Queue()
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

def collect_image_paths(folder_path, existing_images):
    image_paths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if abort_flag.is_set():
                return image_paths
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dirpath, filename)
                if image_path.replace('\\', '/') not in existing_images:
                    image_paths.append(image_path)
    return image_paths

def resize_image(image_array, target_size=(300, 300)):
    # Calculate the aspect ratio of the original image
    h, w, _ = image_array.shape
    target_w, target_h = target_size
    
    # Calculate the scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate the new size maintaining the aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image
    resized_image = cv2.resize(image_array, (new_w, new_h))
    
    # Create a new image with the target size and place the resized image in the center
    final_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    final_image[top:top+new_h, left:left+new_w] = resized_image
    
    return final_image, (scale, top, left)

def draw_face_locations(image_array, face_locations, target_size=(300, 300)):
    # Resize the image and get the scaling parameters
    resized_image, (scale, top, left) = resize_image(image_array, target_size)

    # Convert the resized RGB image array to BGR for OpenCV
    image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    # Draw rectangles around detected faces
    for (top_loc, right_loc, bottom_loc, left_loc) in face_locations:
        # Adjust face locations based on the scaling and offset
        top_loc = int((top_loc * scale + top))
        right_loc = int((right_loc * scale + left))
        bottom_loc = int((bottom_loc* scale + top))
        left_loc = int((left_loc * scale+ left))
        # Draw rectangle on the resized image
        cv2.rectangle(image_bgr, (left_loc, top_loc), (right_loc, bottom_loc), (0, 255, 0), 1   )
    
    return image_bgr
    
def get_face_features(image_array):
    start_time = time.time()
    face_locations = face_recognition.face_locations(image_array)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to find locations: {elapsed_time:.4f} seconds")
    start_time = time.time()
    face_encodings = face_recognition.face_encodings(image_array, face_locations)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken find embedding: {elapsed_time:.4f} seconds")
    return face_encodings, face_locations

def process_faces(image_paths, face_feature_queue, process_progress):
    def process_image(image_path):
        if abort_flag.is_set():
            print(f"Aborting process for {image_path}")
            return
        
        try:
            image_array = face_recognition.load_image_file(image_path)
            face_features, face_locations = get_face_features(image_array)
            if face_features:
                cv2.imshow("output",draw_face_locations(image_array, face_locations))
                cv2.waitKey(0)
                face_feature_queue.put((image_path, face_features, face_locations))
            else:
                face_feature_queue.put((image_path, None, None))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        
        with progress_lock:
            process_progress.update(1)

    for image_path in image_paths:
        process_image(image_path)
        if abort_flag.is_set():
            print("Aborting remaining tasks")
            break

    face_feature_queue.put(None)  # Signal that processing is done
    print("Completed processing of faces")

def write_face_features_to_database(face_feature_queue, db_filename, write_progress):
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                 (filename TEXT, face_index INTEGER, features BLOB, locations BLOB, PRIMARY KEY (filename, face_index))''')

    while True:
        item = face_feature_queue.get()
        if item is None:
            break
        image_path, face_features, face_locations = item
        if face_features is None:
            c.execute("INSERT OR REPLACE INTO faces VALUES (?, ?, NULL, NULL)",
                      (image_path.replace('\\', '/'), 0))
        else:
            for index, (face_feature, face_location) in enumerate(zip(face_features, face_locations)):
                serialized_location = pickle.dumps(face_location)
                c.execute("INSERT OR REPLACE INTO faces VALUES (?, ?, ?, ?)",
                          (image_path.replace('\\', '/'), index, face_feature.tobytes(), serialized_location))
        write_progress.update(1)
        face_feature_queue.task_done()

    conn.commit()
    conn.close()
    print("Writing face features and locations complete, connection closed")

def create_face_features_database(folder_name):
    print(f"Folder is {folder_name}")
    db_filename = 'face_features.db'

    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                 (filename TEXT, face_index INTEGER, features BLOB, locations BLOB, PRIMARY KEY (filename, face_index))''')
    c.execute('SELECT filename FROM faces')
    existing_images = {row[0] for row in c.fetchall()}
    conn.close()

    image_paths = collect_image_paths(folder_name, existing_images)
    total_images_in_directory = len(image_paths)
    print(f"Total new images to process: {total_images_in_directory}")

    process_progress = tqdm.tqdm(total=total_images_in_directory, desc="Processing faces")
    face_write_progress = tqdm.tqdm(total=total_images_in_directory, desc="Writing face features to database")

    process_thread = threading.Thread(target=process_faces, args=(image_paths, face_feature_queue, process_progress))
    face_write_thread = threading.Thread(target=write_face_features_to_database, args=(face_feature_queue, db_filename, face_write_progress))

    process_thread.start()
    face_write_thread.start()

    abort_thread = threading.Thread(target=listen_for_abort, daemon=True)
    abort_thread.start()

    process_thread.join()
    face_write_thread.join()

    print(f"Database updated: {db_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate and store face features independently.')
    parser.add_argument('folder_name', type=str, nargs='?', default='../images', help='Folder name within the default directory to scan for images')
    args = parser.parse_args()

    print("Press 'q' to abort...")    

    folder_name = args.folder_name
    create_face_features_database(folder_name)
    print("Closed main")