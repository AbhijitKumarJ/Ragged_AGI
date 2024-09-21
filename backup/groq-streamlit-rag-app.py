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
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain_groq import ChatGroq

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Set up persistent directory for Chroma
CHROMA_DB_DIR = "chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Set up SQLite database
SQLITE_DB_PATH = "app_data.db"

# Create global Groq client
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# llm_client = OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.environ.get("GROQ_API_KEY")
# )

llm_client = ChatGroq(temperature=0, api_key=os.environ.get("GROQ_API_KEY") , model_name="llama-3.1-70b-versatile")

#Groq(api_key=groq_api_key)

# Streamlit configuration
st.set_page_config(page_title="RAG App with Chroma DB and Groq", layout="wide", initial_sidebar_state="expanded")

# Apply dark theme
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #fafafa;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: #fafafa;
    }
    .stTextInput>div>div>input {
        background: #262730;
        color: #fafafa;
    }
    .stSelectbox>div>div>select {
        background: #262730;
        color: #fafafa;
    }
    .stTextArea>div>div>textarea {
        background: #262730;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
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
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  query_text TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
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

def log_query(query_text):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO queries (query_text) VALUES (?)", (query_text,))
    conn.commit()
    conn.close()

@st.cache_data
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

@st.cache_data
def process_url(url):
    loader = UnstructuredURLLoader(urls=[url])
    documents = loader.load()
    return documents

@st.cache_resource
def create_chroma_db(_documents, collection_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(_documents)

    embeddings = HuggingFaceEmbeddings()
    
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name
    )
    db.persist()
    return db

@st.cache_data
def visualize_user_actions():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT action_type, COUNT(*) as count FROM user_actions GROUP BY action_type", conn)
    conn.close()

    fig = px.bar(df, x='action_type', y='count', title='User Actions')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa'
    )
    return fig

@st.cache_data
def visualize_indexed_files():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT file_type, COUNT(*) as count FROM indexed_files GROUP BY file_type", conn)
    conn.close()

    fig = px.pie(df, values='count', names='file_type', title='Indexed Files by Type')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa'
    )
    return fig

@st.cache_data
def visualize_queries():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT query_text, COUNT(*) as count FROM queries GROUP BY query_text ORDER BY count DESC LIMIT 10", conn)
    conn.close()

    fig = px.bar(df, x='query_text', y='count', title='Top 10 Most Common Queries')
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa'
    )
    return fig

@st.cache_resource
def get_chroma_db(collection_name):
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=collection_name)

def query_chroma_db(query, collection_name):
    db = get_chroma_db(collection_name)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Use Groq client instead of OpenAI
    #llm = OpenAI(temperature=0, client=groq_client)
    qa_chain = RetrievalQA.from_chain_type(llm=llm_client, chain_type="stuff", retriever=retriever)
    
    result = qa_chain.run(query)
    return result

def main():
    init_db()

    st.sidebar.title("RAG App with Chroma DB and Groq")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Add Content", "Query", "Analytics"])

    if app_mode == "Add Content":
        st.header("Add Content to Chroma DB")
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

    elif app_mode == "Query":
        st.header("Query Chroma DB")
        collection_name = st.text_input("Enter the collection name:")
        query = st.text_area("Enter your query:")
        if st.button("Submit Query"):
            if collection_name and query:
                with st.spinner("Processing query..."):
                    result = query_chroma_db(query, collection_name)
                    st.subheader("Query Result:")
                    st.write(result)
                    log_query(query)
                    log_user_action("Query Chroma DB")
            else:
                st.warning("Please enter both a collection name and a query.")

    elif app_mode == "Analytics":
        st.header("Analytics Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("User Actions")
            st.plotly_chart(visualize_user_actions(), use_container_width=True)
        with col2:
            st.subheader("Indexed Files")
            st.plotly_chart(visualize_indexed_files(), use_container_width=True)
        
        st.subheader("Query Analytics")
        st.plotly_chart(visualize_queries(), use_container_width=True)

if __name__ == "__main__":
    main()
