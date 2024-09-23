import os
import time
import json
from flask import Flask, request, Response, stream_with_context
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from collection_manager import get_all_collections
from database_utils import get_chroma_db

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get('GROQ_MODEL', "llama-3.1-8b-instant")
CHROMA_DB_DIR = "chroma_db"

# Chroma DB setup
embeddings = HuggingFaceEmbeddings()

def get_rag_context(query):
    contexts = []
    collections = get_all_collections()
    for collection in collections:
        db = get_chroma_db(collection['id'])
        retriever = db.as_retriever(search_kwargs={"k": 2})
        docs = retriever.get_relevant_documents(query)
        contexts.extend([doc.page_content for doc in docs])
    return "\n\n".join(contexts)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    stream = data.get('stream', False)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Extract the user's query from the messages
    user_query = next((msg['content'] for msg in data.get('messages', []) if msg['role'] == 'user'), "")
    
    # Get RAG context
    rag_context = get_rag_context(user_query)
    
    # Prepare the payload for Groq
    groq_payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": f"Use the following context to answer the user's question:\n\n{rag_context}"},
            *data.get('messages', [])
        ],
        "stream": stream
    }
    
    # Add optional parameters if they're in the request
    for param in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
        if param in data:
            groq_payload[param] = data[param]
    
    if stream:
        return stream_response(groq_payload, headers)
    else:
        return normal_response(groq_payload, headers)

def stream_response(groq_payload, headers):
    def generate():
        response = requests.post(GROQ_API_URL, json=groq_payload, headers=headers, stream=True)
        for line in response.iter_lines():
            if line:
                yield f"{line.decode('utf-8')}\n\n"
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

def normal_response(groq_payload, headers):
    response = requests.post(GROQ_API_URL, json=groq_payload, headers=headers)
    return response.json()

if __name__ == '__main__':
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    app.run(debug=True, host='0.0.0.0', port=5000)