import os
import time
import json
from flask import Flask, request, Response, stream_with_context
import requests
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from collection_manager import get_all_collections
from database_utils import get_chroma_db

app = Flask(__name__)

OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2:1.5b')
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
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    
    # Extract the user's query from the messages
    user_query = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
    
    # Get RAG context
    rag_context = get_rag_context(user_query)
    
    # Add RAG context as a system message
    system_message = {"role": "system", "content": f"Use the following context to answer the user's question:\n\n{rag_context}"}
    messages = [system_message] + messages
    
    # Convert OpenAI format to Ollama format
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    ollama_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream
    }
    
    if stream:
        return stream_response(ollama_payload)
    else:
        return normal_response(ollama_payload)

def stream_response(ollama_payload):
    def generate():
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=ollama_payload, stream=True)
        for line in response.iter_lines():
            if line:
                ollama_chunk = json.loads(line)
                chunk = format_chunk(ollama_chunk)
                yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

def normal_response(ollama_payload):
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=ollama_payload)
    ollama_response = response.json()
    
    return format_response(ollama_response)

def format_chunk(ollama_chunk):
    return {
        "id": "chatcmpl-" + os.urandom(12).hex(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": OLLAMA_MODEL,
        "choices": [{
            "index": 0,
            "delta": {
                "content": ollama_chunk.get('response', '')
            },
            "finish_reason": "stop" if ollama_chunk.get('done', False) else None
        }]
    }

def format_response(ollama_response):
    return {
        "id": "chatcmpl-" + os.urandom(12).hex(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": OLLAMA_MODEL,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ollama_response['response']
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1
        }
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
