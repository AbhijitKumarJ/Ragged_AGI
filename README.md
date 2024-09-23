# Ragged_AGI
This is an attempt on building multipurpose intelligent RAG.

## Till now 3 modules are implemented - 1 pending with initial code:

### 1. RAG Manager: 

This is invoked through ragmanager.py as streamlit app -> 

streamlit run ragmanager.py


### 2. OpenAI compatible Groq wrapper with RAG support:

This is implemented in mini-litellm-groq.py file which can be run directly as a flask app api. This api can be used anywhere,
being openai compatible, i added it to Continue extension of VSCode to leverage the files added to above Rag Manager directly in vscode extension chat.

continue config:

{
      "title": "OpenAI-compatible server / API - Custom",
      "provider": "openai",
      "model": "MODEL_NAME",
      "apiKey": "EMPTY",
      "apiBase": "http://localhost:5000/v1"
}

### 3. Lite llm inspired wrapper for ollama and groq:

This is available in litellmmini folder for your R&D

### 4. OpenAI compatible Ollama wrapper with RAG support:

This just requires adding rag to ollama litellmmini file but is pending till now.
