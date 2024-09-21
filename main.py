import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from file_processors import process_file, process_url
from database_utils import (
    init_db, log_user_action, log_indexed_file, log_query,
    visualize_user_actions, visualize_indexed_files, visualize_queries
)

# Load environment variables
load_dotenv()

# Constants
CHROMA_DB_DIR = "chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Set up Groq client
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

llm_client = ChatGroq(temperature=0, api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

# Streamlit configuration
st.set_page_config(page_title="RAG App with Chroma DB and Groq", layout="wide", initial_sidebar_state="expanded")

# Apply custom styling
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #2C3E50, #4CA1AF);
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background: rgba(44, 62, 80, 0.8);
    }
    .Widget>label {
        color: #FFFFFF;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stButton>button {
        background-color: #3498DB;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    /* Improve visibility of text input */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def create_chroma_db(_documents, collection_name):
    """
    Create a Chroma database from the given documents.
    
    Args:
    _documents (list): List of documents to be added to the database.
    collection_name (str): Name of the collection to be created.
    
    Returns:
    Chroma: The created Chroma database.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(_documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    
    # Create and persist the Chroma database
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name
    )
    db.persist()
    return db

@st.cache_resource
def get_chroma_db(collection_name):
    """
    Retrieve a Chroma database for the given collection name.
    
    Args:
    collection_name (str): Name of the collection to retrieve.
    
    Returns:
    Chroma: The retrieved Chroma database.
    """
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=collection_name)

def query_chroma_db(query, collection_name):
    """
    Query the Chroma database with the given query.
    
    Args:
    query (str): The query to run against the database.
    collection_name (str): Name of the collection to query.
    
    Returns:
    str: The result of the query.
    """
    db = get_chroma_db(collection_name)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm_client, chain_type="stuff", retriever=retriever)
    
    result = qa_chain.run(query)
    return result

def get_available_collections():
    """
    Get a list of available collections in the Chroma database.
    
    Returns:
    list: List of available collection names.
    """
    collections = []
    for item in os.listdir(CHROMA_DB_DIR):
        if os.path.isdir(os.path.join(CHROMA_DB_DIR, item)):
            collections.append(item)
    return collections

def delete_collection(collection_name):
    """
    Delete a collection from the Chroma database.
    
    Args:
    collection_name (str): Name of the collection to delete.
    """
    db = get_chroma_db(collection_name)
    db.delete_collection()
    st.success(f"Collection '{collection_name}' deleted successfully.")

def export_collection(collection_name):
    """
    Export the contents of a collection as a JSON string.
    
    Args:
    collection_name (str): Name of the collection to export.
    
    Returns:
    str: JSON string containing the exported collection data.
    """
    db = get_chroma_db(collection_name)
    documents = db.get()
    
    export_data = []
    for doc in documents['documents']:
        export_data.append({
            'content': doc,
            'metadata': documents['metadatas'][documents['documents'].index(doc)]
        })
    
    return json.dumps(export_data, indent=2)

def query_without_rag(query):
    """
    Query the LLM directly without using RAG.
    
    Args:
    query (str): The query to send to the LLM.
    
    Returns:
    str: The response from the LLM.
    """
    response = llm_client.predict(query)
    return response

def main():
    """
    Main function to run the Streamlit application.
    """
    init_db()

    st.sidebar.title("RAG App with Chroma DB and Groq")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Add Content", "Query", "Manage Collections", "Analytics"])

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
        collections = get_available_collections()
        collection_name = st.selectbox("Select a collection:", collections)
        query = st.text_area("Enter your query:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Query"):
                if collection_name and query:
                    with st.spinner("Processing query..."):
                        result = query_chroma_db(query, collection_name)
                        st.subheader("Query Result (with RAG):")
                        st.write(result)
                        log_query(query)
                        log_user_action("Query Chroma DB")
                else:
                    st.warning("Please select a collection and enter a query.")
        
        with col2:
            if st.button("Compare (With vs Without RAG)"):
                if collection_name and query:
                    with st.spinner("Processing query..."):
                        rag_result = query_chroma_db(query, collection_name)
                        non_rag_result = query_without_rag(query)
                        
                        st.subheader("Query Result (with RAG):")
                        st.write(rag_result)
                        
                        st.subheader("Query Result (without RAG):")
                        st.write(non_rag_result)
                        
                        log_query(query)
                        log_user_action("Compare Query")
                else:
                    st.warning("Please select a collection and enter a query.")

    elif app_mode == "Manage Collections":
        st.header("Manage Collections")
        collections = get_available_collections()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Delete Collection")
            collection_to_delete = st.selectbox("Select a collection to delete:", collections)
            if st.button("Delete Collection"):
                if st.checkbox("I understand this action is irreversible"):
                    delete_collection(collection_to_delete)
                    st.experimental_rerun()
                else:
                    st.warning("Please confirm that you understand this action is irreversible.")
        
        with col2:
            st.subheader("Export Collection")
            collection_to_export = st.selectbox("Select a collection to export:", collections)
            if st.button("Export Collection"):
                exported_data = export_collection(collection_to_export)
                st.download_button(
                    label="Download Export",
                    data=exported_data,
                    file_name=f"{collection_to_export}_export.json",
                    mime="application/json"
                )

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