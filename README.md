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

This is implemented in mini-litellm-ollama.py file which can be run directly as a flask app api like groq one above. Note: this needs further testing.


# AIM of this project:

## 1. Server RAG or Model independent knowledge system:

This has many implications:

a. If a separate dedicated endpoint is created for RAG based llm inference, then there is no limit on cut off date or capacity of model. Model's knowledgebase will not be out of date atleast for quite sometime since rag system can be constantly updated.

b. Knowledge rag system can be divided and standardized so that people ca download and use it if required in local environments also.

c. Since it will be centralized, improvement upon the knowledge base as well as far advanced alogoriths and infrastructure can be applied.

d. If required, it can be monetized with a bit higher cost as having a db is far easy and less costly than training a model to perfection on same knowledge.

5. With openai compatible endpoint it can be directly used in tools and extensions like continue, aider and others

## 2. Local RAG scaled openai compatible inference:

Benefits should be obvious:
a. It can be optimized for personal data across all ai based app running locally.
b. With openai compatible endpoint it can be directly used in tools and extensions like continue, aider and others
c. People can add their own variations/improvements.
