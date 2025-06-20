import uuid
from typing import Any, Callable, Dict, List, Optional

import chromadb


class ChromaClient:
    """
    Class for managing ChromaDB collections, including creating, updating and deleting collections.
    """

    def __init__(self, chroma_host: str, chroma_port: int, embedder: Callable):
        """
        Initialize ChromaDB client with connection parameters.

        Args:
            chroma_host: Host address of the ChromaDB server
            chroma_port: Port number of the ChromaDB server
            embedder: Function to use for generating embeddings
        """
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        self.embedder = embedder

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        collections = self.chroma_client.list_collections()
        return any(col == collection_name for col in collections)

    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Delete a collection if it exists.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            Dictionary containing success status and message
        """
        result = {"success": False, "message": ""}

        try:
            if not self.collection_exists(collection_name):
                result["success"] = True
                result["message"] = f"Collection '{collection_name}' does not exist."
                return result

            self.chroma_client.delete_collection(name=collection_name)
            result["success"] = True
            result["message"] = f"Collection '{collection_name}' successfully deleted."

        except ValueError as e:
            result["message"] = f"Validation error: {str(e)}"
        except Exception as e:
            result["message"] = f"Unexpected error: {str(e)}"

        return result

    def update(
        self,
        collection_name: str,
        embeddings: List[Any],
        documents: List[Any],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Update a collection by recreating it with new data.

        Args:
            collection_name: Name of the collection to update
            embeddings: List of embeddings to add to the collection
            documents: List of documents to add to the collection
            metadatas: Optional list of metadata for each document

        Returns:
            Dictionary containing success status and message
        """
        result = {"success": False, "message": ""}

        try:
            self._validate_inputs(collection_name, embeddings, documents, metadatas)

            if self.collection_exists(collection_name):
                delete_result = self.delete_collection(collection_name)
                if not delete_result["success"]:
                    return delete_result

            collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedder,
                metadata={"hnsw:space": "cosine"},
            )

            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=[document.get("page_content") for document in documents],
            )

            result["success"] = True
            result["message"] = (
                f"Collection '{collection_name}' successfully updated with {len(documents)} documents."
            )

        except ValueError as e:
            result["message"] = f"Validation error: {str(e)}"
        except Exception as e:
            result["message"] = f"Unexpected error: {str(e)}"

        return result

    def _validate_inputs(
        self,
        collection_name: str,
        embeddings: List[Any],
        documents: List[Any],
        metadatas: Optional[List[Dict[str, Any]]],
    ) -> None:
        """
        Validate input parameters for collection operations.

        Args:
            collection_name: Name of the collection
            embeddings: List of embeddings to add
            documents: List of documents to add
            metadatas: Optional list of metadata for each document

        Raises:
            ValueError: If any validation check fails
        """
        if not collection_name or not isinstance(collection_name, str):
            raise ValueError("Collection name must be a non-empty string")
        if not documents or not isinstance(documents, list):
            raise ValueError("Documents must be a non-empty list")

        if not embeddings or not isinstance(embeddings, list):
            raise ValueError("Embeddings must be a non-empty list")

        if metadatas and len(metadatas) != len(embeddings):
            raise ValueError("Number of metadata items must match number of embeddings")
