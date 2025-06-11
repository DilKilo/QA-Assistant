from typing import Any, Dict, List, Optional

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


class VertexAIChromaEmbedder(EmbeddingFunction[Documents]):
    """
    A custom embedding function for ChromaDB using Vertex AI TextEmbeddingModel.
    Follows ChromaDB's EmbeddingFunction interface for better compatibility.
    """

    def __init__(
        self,
        model_name: str,
        task_type: str,
        dimensions: Optional[int] = None,
        batch_size: int = 5,
        retry_attempts: int = 3,
    ) -> None:
        """
        Initializes the embedder based on Vertex AI.

        Args:
            model_name: Name of the pretrained embedding model.
            task_type: Type of task for embedding optimization:
                - "RETRIEVAL_DOCUMENT": for indexed documents
                - "RETRIEVAL_QUERY": for search queries
                - "QUESTION_ANSWERING": for question answering tasks
                - "SEMANTIC_SIMILARITY": for semantic similarity tasks
                - "CLASSIFICATION": for classification tasks
                - "CLUSTERING": for clustering tasks
            dimensions: Target dimensionality of output embeddings. If None,
                       the model's default dimensionality is used.
            batch_size: Batch size for processing texts.
            retry_attempts: Number of retry attempts for API errors.
        """
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.task_type = task_type
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts

    def __call__(self, input: Documents) -> Embeddings:
        """
        Creates embeddings for a list of documents.
        This follows ChromaDB's EmbeddingFunction interface.

        Args:
            input: List of strings (documents) to embed.

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        for i in range(0, len(input), self.batch_size):
            batch_texts = input[i : i + self.batch_size]
            batch_embeddings = self._get_embeddings_with_retry(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _get_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Gets embeddings with retry support for error handling.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            Exception: If all embedding attempts fail.
        """
        attempts = 0
        last_error = None

        while attempts < self.retry_attempts:
            try:
                return self._get_embeddings_batch(texts)
            except Exception as e:
                attempts += 1
                last_error = e

        raise Exception(
            f"Failed to get embeddings after {self.retry_attempts} attempts: {last_error}"
        )

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Creates embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        inputs = [
            TextEmbeddingInput(text=text, task_type=self.task_type) for text in texts
        ]

        kwargs: Dict[str, Any] = {}
        if self.dimensions is not None:
            kwargs["output_dimensionality"] = self.dimensions

        embeddings = self.model.get_embeddings(inputs, **kwargs)

        embedding_vectors = [embedding.values for embedding in embeddings]

        return embedding_vectors

    def get_dimensions(self) -> int:
        """
        Returns the dimensionality of embeddings.

        Returns:
            Embedding dimensionality (integer).
        """
        if self.dimensions is not None:
            return self.dimensions

        sample_embedding = self._get_embeddings_batch(["Sample text"])[0]
        return len(sample_embedding)


class VertexAITokenizer:
    """
    A tokenizer class that uses Vertex AI TextEmbeddingModel to count tokens in text.
    Provides a callable interface compatible with token counting requirements.
    """

    def __init__(self, model_name: str, task_type: str) -> None:
        """
        Initializes the tokenizer based on Vertex AI.

        Args:
            model_name: Name of the pretrained embedding model.
            task_type: Type of task for embedding optimization:
                - "RETRIEVAL_DOCUMENT": for indexed documents
                - "RETRIEVAL_QUERY": for search queries
                - "QUESTION_ANSWERING": for question answering tasks
                - "SEMANTIC_SIMILARITY": for semantic similarity tasks
                - "CLASSIFICATION": for classification tasks
                - "CLUSTERING": for clustering tasks
        """
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.task_type = task_type

    def __call__(self, text: str) -> int:
        """
        Counts tokens in the provided text using Vertex AI's model statistics.
        This method allows the tokenizer to be used as a callable function.

        Args:
            text: String text to tokenize and count.

        Returns:
            Number of tokens in the input text according to the model.
        """
        inputs = [TextEmbeddingInput(text=text, task_type=self.task_type)]
        embeddings = self.model.get_embeddings(inputs)
        return embeddings[0].statistics.token_count
