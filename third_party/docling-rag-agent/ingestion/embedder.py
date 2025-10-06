"""
Document embedding generation for vector search.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from .chunker import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for document chunks using a local SentenceTransformer model."""

    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v2-base-en'):
        """
        Initialize embedding generator.

        Args:
            model_name: The name of the SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks.

        Returns:
            Chunks with embeddings added.
        """
        if not chunks:
            return []

        contents = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(contents, show_progress_bar=True)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()

        return chunks

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query.

        Returns:
            Query embedding.
        """
        embedding = self.model.encode(query)
        return embedding.tolist()

_EMBEDDER: EmbeddingGenerator | None = None

def create_embedder() -> EmbeddingGenerator:
    """Return a cached embedding generator instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = EmbeddingGenerator()
    return _EMBEDDER
