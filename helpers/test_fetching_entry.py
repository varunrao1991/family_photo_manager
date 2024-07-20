import sqlite3
import os

# Connect to the SQLite database
connection = sqlite3.connect('image_similarity.db')

# Create a cursor object
cursor = connection.cursor()

image_req = "DCIM/Images_cam/20201201_142418.jpg"

file_path = os.path.join("F:/",image_req)
# Example query
cursor.execute("SELECT * FROM images WHERE filename = ?", (file_path,))

# Fetch and print all rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
connection.close()
