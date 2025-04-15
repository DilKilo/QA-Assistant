# ChromaDB
COLLECTION_NAME = "collection_text-embedding-005"
CHROMA_HOST = ...
CHROMA_PORT = ...
RETRIEVAL_RESULTS = 5

# Vertex AI embedding model
VERTEXAI_MODEL_EMBEDDING_NAME = "text-embedding-005"
# Use "QUESTION_ANSWERING" or "RETRIEVAL_QUERY"
VERTEXAI_TASK_TYPE = "QUESTION_ANSWERING"
VECTOR_DIMENSIONS = 512
GENERATION_CONFIG = {
    "temperature": 0.2,
    "max_output_tokens": 1024,
    "top_p": 0.8,
    "top_k": 40
}

# Vertex AI model
VERTEXAI_MODEL_NAME = "gemini-1.5-flash"


# Google Chat
GOOGLE_CHAT_SCOPES = ['https://www.googleapis.com/auth/chat.bot']
