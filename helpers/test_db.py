import sqlite3

# Connect to the SQLite database
connection = sqlite3.connect('image_similarity.db')

# Create a cursor object
cursor = connection.cursor()

# Example query
cursor.execute("SELECT filename FROM images")

# Fetch and print all rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
connection.close()
