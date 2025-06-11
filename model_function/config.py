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
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The model answer to the question based on the provided context.",
        },
        "sources_used": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "The list of document numbers used to answer the question, starting from 1.",
        },
        "answer_language": {
            "type": "string",
            "description": "The language of the answer in ISO 639-1 format.",
        },
    },
    "required": ["answer", "sources_used", "answer_language"],
}

GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.8,
    "response_mime_type": "application/json",
    "response_schema": _RESPONSE_SCHEMA,
}

# Vertex AI model
VERTEXAI_MODEL_NAME = "gemini-1.5-flash"


# Google Chat
GOOGLE_CHAT_SCOPES = ["https://www.googleapis.com/auth/chat.bot"]
