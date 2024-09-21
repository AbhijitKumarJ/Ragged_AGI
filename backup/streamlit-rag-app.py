import streamlit as st
import requests
import tempfile
import os
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredURLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Set up persistent directory for Chroma
CHROMA_DB_DIR = "chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Set up SQLite database
SQLITE_DB_PATH = "app_data.db"

def init_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_actions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  action_type TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS indexed_files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  file_name TEXT,
                  file_type TEXT,
                  file_size INTEGER,
                  indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def log_user_action(action_type):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO user_actions (action_type) VALUES (?)", (action_type,))
    conn.commit()
    conn.close()

def log_indexed_file(file_name, file_type, file_size):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO indexed_files (file_name, file_type, file_size) VALUES (?, ?, ?)",
              (file_name, file_type, file_size))
    conn.commit()
    conn.close()

def process_file(file, file_type):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
        temp_file.write(file.getvalue())
        file_path = temp_file.name

    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type in ['docx', 'doc']:
        loader = Docx2txtLoader(file_path)
    elif file_type == 'txt':
        loader = TextLoader(file_path)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None

    documents = loader.load()
    os.unlink(file_path)
    return documents

def process_url(url):
    loader = UnstructuredURLLoader(urls=[url])
    documents = loader.load()
    return documents

def create_chroma_db(documents, collection_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name
    )
    db.persist()
    return db

def visualize_user_actions():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT action_type, COUNT(*) as count FROM user_actions GROUP BY action_type", conn)
    conn.close()

    fig = px.bar(df, x='action_type', y='count', title='User Actions')
    st.plotly_chart(fig)

def visualize_indexed_files():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT file_type, COUNT(*) as count FROM indexed_files GROUP BY file_type", conn)
    conn.close()

    fig = px.pie(df, values='count', names='file_type', title='Indexed Files by Type')
    st.plotly_chart(fig)

def main():
    st.title("RAG App with Chroma DB")

    init_db()

    tab1, tab2, tab3 = st.tabs(["Add Content", "User Actions", "Indexed Files"])

    with tab1:
        input_type = st.radio("Select input type:", ("File", "URL"))

        if input_type == "File":
            uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'doc', 'txt'])
            if uploaded_file:
                file_type = uploaded_file.name.split('.')[-1].lower()
                documents = process_file(uploaded_file, file_type)
                if documents:
                    collection_name = f"file_{uploaded_file.name}"
                    metadata = {
                        "file_name": uploaded_file.name,
                        "file_type": file_type,
                        "file_size": uploaded_file.size
                    }
                    log_user_action("File Upload")
                else:
                    st.error("Failed to process the file.")
                    return
        else:
            url = st.text_input("Enter a URL:")
            if url:
                documents = process_url(url)
                collection_name = f"url_{hash(url)}"
                metadata = {
                    "source_url": url
                }
                log_user_action("URL Process")
            else:
                st.warning("Please enter a URL.")
                return

        if st.button("Create Chroma DB"):
            if documents:
                with st.spinner("Creating Chroma DB..."):
                    db = create_chroma_db(documents, collection_name)
                    st.success(f"Chroma DB created successfully! Collection name: {collection_name}")
                    st.json(metadata)
                    if input_type == "File":
                        log_indexed_file(metadata['file_name'], metadata['file_type'], metadata['file_size'])
                    log_user_action("Create Chroma DB")
            else:
                st.error("No documents to process. Please upload a file or enter a valid URL.")

    with tab2:
        st.header("User Actions Visualization")
        visualize_user_actions()

    with tab3:
        st.header("Indexed Files Visualization")
        visualize_indexed_files()

if __name__ == "__main__":
    main()
