import sqlite3
import os

# Constants
DB_NAME = "collections.db"

def init_db():
    """Initialize the SQLite database and create the collections table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS collections
                 (id TEXT PRIMARY KEY,
                  name TEXT,
                  file_name TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def add_collection(collection_id, collection_name, file_name):
    """Add a new collection to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO collections (id, name, file_name) VALUES (?, ?, ?)",
              (collection_id, collection_name, file_name))
    conn.commit()
    conn.close()

def get_collection_info(collection_id):
    """Retrieve information about a specific collection."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return {'id': result[0], 'name': result[1], 'file_name': result[2], 'created_at': result[3]}
    return None

def get_all_collections():
    """Retrieve information about all collections."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM collections")
    results = c.fetchall()
    conn.close()
    return [{'id': r[0], 'name': r[1], 'file_name': r[2], 'created_at': r[3]} for r in results]

def delete_collection(collection_id):
    """Delete a collection from the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
    conn.commit()
    conn.close()

def update_collection_name(collection_id, new_name):
    """Update the name of a collection."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE collections SET name = ? WHERE id = ?", (new_name, collection_id))
    conn.commit()
    conn.close()
