import sqlite3
import os
import argparse

# Database file path
DATABASE = './tmp/image_features.db'

def normalize_path(path):
    # Normalize and convert to forward slashes
    return os.path.normpath(path).replace("\\", "/")

def standardize_path(path, base_folder):
    path = normalize_path(path)
    base_folder = normalize_path(base_folder)

    # Remove base folder prefix if present
    if path.startswith(base_folder + "/"):
        return path[len(base_folder) + 1:]
    return path

def update_database(base_folder):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT filename FROM images")
    rows = cursor.fetchall()

    seen = set()

    for row in rows:
        old_path = row[0]
        new_path = standardize_path(old_path, base_folder)

        if new_path in seen:
            print(f"Duplicate found: {old_path} -> matches existing {new_path}, removing...")
            cursor.execute("DELETE FROM images WHERE filename = ?", (old_path,))
        elif old_path != new_path:
            try:
                cursor.execute("UPDATE images SET filename = ? WHERE filename = ?", (new_path, old_path))
                print(f"Updated: {old_path} -> {new_path}")
                seen.add(new_path)
            except sqlite3.IntegrityError:
                print(f"Conflict detected: {new_path} already exists. Removing {old_path}")
                cursor.execute("DELETE FROM images WHERE filename = ?", (old_path,))
        else:
            seen.add(old_path)

    conn.commit()
    conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean and standardize image paths in SQLite DB.")
    parser.add_argument('--root', type=str, required=True, help='Root image folder to trim from paths (e.g., D:/images)')
    args = parser.parse_args()

    update_database(args.root)
