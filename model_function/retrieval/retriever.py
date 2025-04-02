from typing import Dict, List, Any, Optional
import chromadb
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


class ChromaRetriever:
    """
    A class for working with ChromaDB to retrieve and format relevant documents.

    This class provides an interface to connect to a ChromaDB instance,
    retrieve documents based on queries, and format the results for use
    in RAG (Retrieval Augmented Generation) systems.
    """

    def __init__(self,
                 host: str,
                 port: int,
                 collection_name: str,
                 embedding_function: Any) -> None:
        """
        Initialize the ChromaRetriever with connection parameters.

        Args:
            host: ChromaDB server host address
            port: ChromaDB server port
            collection_name: Name of the ChromaDB collection to use
            embedding_function: Function used to create embeddings for queries
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.client = None
        self.collection = None

        self._connect()

    def _connect(self) -> None:
        """
        Establish connection to ChromaDB and retrieve the specified collection.
        """
        try:
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port
            )

            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to ChromaDB: {str(e)}")

    def retrieve(self,
                 query: str,
                 n_results: int = 5,
                 where: Optional[Dict[str, Any]] = None,
                 where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents from ChromaDB based on the query.

        Args:
            query: Text query to search for
            n_results: Number of results to retrieve
            where: Optional filter for metadata
            where_document: Optional filter for document content

        Returns:
            Dictionary containing query results from ChromaDB
        """
        try:
            query_params = {
                "query_texts": [query],
                "n_results": n_results
            }

            if where:
                query_params["where"] = where

            if where_document:
                query_params["where_document"] = where_document

            results = self.collection.query(**query_params)
            return results

        except Exception as e:
            raise Exception(f"Failed to retrieve documents: {str(e)}")

    def format_context(self, retrieval_result: Dict[str, Any], join_str: str = "\n\n") -> str:
        """
        Format retrieved documents into a single context string.

        Args:
            retrieval_result: Results from the retrieve method
            join_str: String used to join multiple documents

        Returns:
            Formatted context string for use in prompts
        """
        if not retrieval_result or "documents" not in retrieval_result or not retrieval_result["documents"]:
            print("No documents for context formatting")
            return ""

        documents = retrieval_result["documents"][0]

        context = join_str.join(documents)
        context = context.replace("'", "").replace('"', "")

        return context
