import os
import json
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

CHROMA_DB_DIR = "chroma_db"

def create_chroma_db(_documents, collection_name):
    """
    Create a Chroma database from the given documents.
    """
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

def get_chroma_db1():
    """
    Retrieve a Chroma database for the given collection name.
    """
    embeddings = HuggingFaceEmbeddings()
    chro= Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name="collections")
    print(chro.get())


def get_chroma_db(collection_name):
    """
    Retrieve a Chroma database for the given collection name.
    """
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=collection_name)

def query_chroma_db(query, collection_name):
    """
    Query the Chroma database with the given query.
    """
    #get_chroma_db1()
    db = get_chroma_db(collection_name)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    llm_client = ChatGroq(temperature=0, api_key=os.environ.get("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")
    qa_chain = RetrievalQA.from_chain_type(llm=llm_client, chain_type="stuff", retriever=retriever)
    
    result = qa_chain.run(query)
    return result

def get_available_collections():
    """
    Get a list of available collections in the Chroma database.
    """
    collections = []
    for item in os.listdir(CHROMA_DB_DIR):
        if os.path.isdir(os.path.join(CHROMA_DB_DIR, item)):
            collections.append({"id": item, "name": item})
    return collections

def delete_collection(collection_name):
    """
    Delete a collection from the Chroma database.
    """
    db = get_chroma_db(collection_name)
    db.delete_collection()

def export_collection(collection_name):
    """
    Export the contents of a collection as a JSON string.
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