import tempfile
import os
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredURLLoader,
    UnstructuredMarkdownLoader
)
import streamlit as st

@st.cache_data
def process_file(file, file_type):
    """
    Process an uploaded file and return the extracted documents.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
        temp_file.write(file.getvalue())
        file_path = temp_file.name

    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type in ['docx', 'doc']:
        loader = Docx2txtLoader(file_path)
    elif file_type == 'txt':
        loader = TextLoader(file_path)
    elif file_type in ['md', 'markdown']:
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None

    documents = loader.load()
    os.unlink(file_path)
    return documents

@st.cache_data
def process_url(url):
    """
    Process a URL and return the extracted documents.
    """
    loader = UnstructuredURLLoader(urls=[url])
    documents = loader.load()
    return documents
