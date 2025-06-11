import os
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ChromaDB
# COLLECTION_NAME = collection_MODEL-NAME_VECTOR-DIMENSIONS_CHUNK-SIZE
COLLECTION_NAME = "collection_text-embedding-005_512_512"
CHROMA_HOST = ...
CHROMA_PORT = ...

# GCP Storage
CHROMA_STORAGE_NAME = "chromadb-vectors-storage"
BACKUP_STORAGE_NAME = "chromadb-backups"

# Backups settings
BACKUPS_NUMBER = 3

# Fetching settings
MAX_WORKERS = 5

# Confluence
CONFLUENCE_URL = "https://innowise-group.atlassian.net"
CONFLUENCE_USERNAME = os.environ.get("CONFLUENCE_USERNAME")
CONFLUENCE_PASSWORD = os.environ.get("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE = "QD"
EXCLUDE_PAGES_IDS = ["2639986781", "2695463266", "2710208755", "2760704026"]

# Chunking settings
KEEP_TAGS = {
    "table",
    "tr",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "a",
    "ol",
    "ul",
    "li",
    "ac:link",
    "ri:user",
    "ri:page",
}
CHUNK_OVERLAP = 0.0
CHUNK_SIZE = 512

# Vertex AI
VERTEXAI_MODEL_NAME = "text-embedding-005"
VERTEXAI_TASK_TYPE = "RETRIEVAL_DOCUMENT"
VERTEXAI_VECTOR_DIMENSIONS = 512
