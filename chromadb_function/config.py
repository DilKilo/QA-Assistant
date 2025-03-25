import os

# ChromaDB
COLLECTION_NAME = "Collection_1"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# GCP Storage
CHROMA_STORAGE_NAME = "chromadb-vectors-storage"
BACKUP_STORAGE_NAME = "chromadb-backups"

# Backups settings
BACKUPS_NUMBER = 3

# Fetching settings
MAX_WORKERS = 5

# Confluence
CONFLUENCE_URL = "https://innowise-group.atlassian.net"
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_PASSWORD = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE = "QD"
EXCLUDE_PAGES_IDS = ['2639986781', '2695463266', '2710208755', '2760704026']

# Chunking settings
KEEP_TAGS = {"table", "tr", "td", "th", "h1", "h2", "h3", "h4",
             "h5", "h6", "a", "ol", "ul", "li", 'ac:link', "ri:user", "ri:page"}
OVERLAP = 0.0
VECTOR_DIMENSIONS = 1024


# Vertex AI
VERTEXAI_MODEL_NAME = "text-embedding-005"
VERTEXAI_TASK_TYPE = "RETRIEVAL_DOCUMENT"
