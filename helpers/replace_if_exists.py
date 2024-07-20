import sqlite3
import os

# Database file path
DATABASE = 'image_similarity.db'

def standardize_path(path):
    # Convert Windows paths to Unix-style paths and remove redundant backslashes
    return path.replace('\\', '/')

def update_database():
    # Connect to the SQLite database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Get all rows from the images table
    cursor.execute("SELECT filename FROM images")
    rows = cursor.fetchall()
    
    for row in rows:
        old_path = row[0]
        new_path = standardize_path(old_path)
        
        if old_path != new_path:
            try:
                # Update the filename with the new standardized path
                cursor.execute("UPDATE images SET filename = ? WHERE filename = ?", (new_path, old_path))
                print(f"Updated: {old_path} -> {new_path}")
            except sqlite3.IntegrityError:
                # If there is a unique constraint error, remove the conflicting entry
                print(f"Conflict detected for: {new_path}. Removing existing entry.")
                cursor.execute("DELETE FROM images WHERE filename = ?", (new_path,))
                # Retry updating the path after removing the existing entry
                cursor.execute("UPDATE images SET filename = ? WHERE filename = ?", (new_path, old_path))
                print(f"Updated after conflict resolution: {old_path} -> {new_path}")
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()

if __name__ == '__main__':
    update_database()
