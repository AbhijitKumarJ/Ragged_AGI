import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from file_processors import process_file, process_url
from database_utils import create_chroma_db, get_chroma_db, query_chroma_db, get_available_collections, delete_collection as delete_chroma_collection, export_collection
from collection_manager import init_db, add_collection, get_all_collections, delete_collection as delete_sqlite_collection, update_collection_name
import nltk

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(
    page_title="RAG App with Chroma DB and Groq",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')

download_nltk_data()

# Initialize SQLite database
init_db()

# Constants
CHROMA_DB_DIR = "chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Set up Groq client
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

llm_client = ChatGroq(temperature=0, api_key=groq_api_key, model_name="llama-3.1-8b-instant")



def query_without_rag(query):
    """
    Query the LLM directly without using RAG.
    """
    response = llm_client.predict(query)
    return response

def main():
    """
    Main function to run the Streamlit application.
    """
    st.sidebar.title("RAG App with Chroma DB and Groq")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Add Content", "Query", "Manage Collections"])

    if app_mode == "Add Content":
        st.header("Add Content to Chroma DB")
        input_type = st.radio("Select input type:", ("File", "URL"))

        if input_type == "File":
            uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'doc', 'txt', 'md', 'markdown'])
            if uploaded_file:
                file_type = uploaded_file.name.split('.')[-1].lower()
                documents = process_file(uploaded_file, file_type)
                if documents:
                    collection_name = st.text_input("Enter a name for this collection:", value=f"file_{uploaded_file.name}")
                    file_name = uploaded_file.name
                else:
                    st.error("Failed to process the file.")
                    return
        else:
            url = st.text_input("Enter a URL:")
            if url:
                documents = process_url(url)
                collection_name = st.text_input("Enter a name for this collection:", value=f"url_{url.split('/')[-1]}")
                file_name = url
            else:
                st.warning("Please enter a URL.")
                return

        if st.button("Create Chroma DB"):
            if documents and collection_name:
                with st.spinner("Creating Chroma DB..."):
                    db = create_chroma_db(documents, collection_name)
                    add_collection(collection_name, collection_name, file_name)
                    st.success(f"Chroma DB created successfully! Collection name: {collection_name}")
            else:
                st.error("No documents to process or collection name not provided. Please upload a file/enter a URL and provide a collection name.")

    elif app_mode == "Query":
        st.header("Query Chroma DB")
        collections = get_all_collections()
        if not collections:
            st.warning("No collections available. Please add content first.")
        else:
            selected_collection = st.selectbox("Select a collection:", [f"{c['name']} ({c['file_name']})" for c in collections])
            collection_id = next(c["id"] for c in collections if f"{c['name']} ({c['file_name']})" == selected_collection)
            
            query = st.text_area("Enter your query:", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Query"):
                    if collection_id and query:
                        with st.spinner("Processing query..."):
                            result = query_chroma_db(query, collection_id)
                            st.subheader("Query Result (with RAG):")
                            st.write(result)
                    else:
                        st.warning("Please select a collection and enter a query.")
            
            with col2:
                if st.button("Compare (With vs Without RAG)"):
                    if collection_id and query:
                        with st.spinner("Processing query..."):
                            rag_result = query_chroma_db(query, collection_id)
                            non_rag_result = query_without_rag(query)
                            
                            st.subheader("Query Result (with RAG):")
                            st.write(rag_result)
                            
                            st.subheader("Query Result (without RAG):")
                            st.write(non_rag_result)
                    else:
                        st.warning("Please select a collection and enter a query.")

    elif app_mode == "Manage Collections":
        st.header("Manage Collections")
        collections = get_all_collections()
        if not collections:
            st.warning("No collections available. Please add content first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Delete Collection")
                collection_to_delete = st.selectbox("Select a collection to delete:", [f"{c['name']} ({c['file_name']})" for c in collections])
                if st.button("Delete Collection"):
                    if st.checkbox("I understand this action is irreversible"):
                        collection_id = next(c["id"] for c in collections if f"{c['name']} ({c['file_name']})" == collection_to_delete)
                        delete_chroma_collection(collection_id)
                        delete_sqlite_collection(collection_id)
                        st.success(f"Collection '{collection_to_delete}' deleted successfully.")
                        st.experimental_rerun()
                    else:
                        st.warning("Please confirm that you understand this action is irreversible.")
            
            with col2:
                st.subheader("Export Collection")
                collection_to_export = st.selectbox("Select a collection to export:", [f"{c['name']} ({c['file_name']})" for c in collections])
                if st.button("Export Collection"):
                    collection_id = next(c["id"] for c in collections if f"{c['name']} ({c['file_name']})" == collection_to_export)
                    exported_data = export_collection(collection_id)
                    st.download_button(
                        label="Download Export",
                        data=exported_data,
                        file_name=f"{collection_to_export}_export.json",
                        mime="application/json"
                    )

            st.subheader("Rename Collection")
            collection_to_rename = st.selectbox("Select a collection to rename:", [f"{c['name']} ({c['file_name']})" for c in collections])
            new_name = st.text_input("Enter new name:")
            if st.button("Rename Collection"):
                collection_id = next(c["id"] for c in collections if f"{c['name']} ({c['file_name']})" == collection_to_rename)
                update_collection_name(collection_id, new_name)
                st.success(f"Collection renamed to '{new_name}' successfully.")
                st.experimental_rerun()

if __name__ == "__main__":
    main()