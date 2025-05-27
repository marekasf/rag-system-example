"""
RetrievalSystem module for retrieving relevant document chunks based on user queries.
This module integrates with VectorStore to implement semantic search and simple re-ranking
capabilities for document retrieval in a RAG system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

# Core document type
from langchain_core.documents import Document

# Import our VectorStore class
from amk.storage.faiss_storage import VectorStore
from amk.processing.embedding_generator import EmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RetrievalController:
    """
    A class for retrieving relevant document chunks based on user queries.

    This class handles:
    - Semantic search using vector embeddings
    - Simple re-ranking based on relevance to the query
    - Basic metadata filtering
    """

    def __init__(
            self,
            vector_store: VectorStore,
            embedding_generator: EmbeddingGenerator,
            top_k: int = 5
    ):
        """
        Initialize the RetrievalSystem with specified components.

        Args:
            vector_store: VectorStore instance for semantic search
            embedding_generator: EmbeddingGenerator for embedding queries
            top_k: Number of results to retrieve
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k

        logger.info(f"Initialized RetrievalSystem with top_k={top_k}")

    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a given query.

        Args:
            query: User query string
            top_k: Number of results to retrieve (overrides instance setting)
            filter_metadata: Optional metadata for filtering results

        Returns:
            List of relevant Document objects
        """
        if top_k is None:
            top_k = self.top_k

        logger.info(f"Retrieving documents for query: '{query}'")

        # Perform semantic search
        semantic_results = self._semantic_search(query, top_k, filter_metadata)

        # TODO: Implement hybrid search (combining semantic and keyword search)
        # Hybrid search would:
        # 1. Perform keyword-based search
        # 2. Combine results with semantic search
        # 3. Remove duplicates

        # Apply simple re-ranking
        reranked_results = self._simple_rerank(query, semantic_results)

        # Extract just the Document objects from the results
        documents = [doc for doc, _ in reranked_results]

        logger.info(f"Retrieved {len(documents)} documents for query")

        return documents

    def _semantic_search(
            self,
            query: str,
            top_k: int,
            filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform semantic search using query embeddings.

        Args:
            query: User query string
            top_k: Number of results to retrieve
            filter_metadata: Optional metadata for filtering results

        Returns:
            List of tuples containing Document objects and similarity scores
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.embed_query(query)

        # Perform similarity search in vector store
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            filter_metadata=filter_metadata
        )

        logger.info(f"Semantic search returned {len(results)} results")

        return results

    def _simple_rerank(
            self,
            query: str,
            results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Apply a simple re-ranking to the search results based on keyword matches.

        Args:
            query: Original query string
            results: List of (document, score) tuples from semantic search

        Returns:
            Re-ranked list of (document, score) tuples
        """
        # Simple re-ranking: boost documents that contain exact phrases from the query
        query_terms = query.lower().split()

        reranked_results = []

        for doc, score in results:
            content = doc.page_content.lower()

            # Count exact matches of query terms in the content
            term_matches = sum(1 for term in query_terms if term in content)

            # Adjust score based on term matches (simple boost)
            # Original semantic score is in range [0,1], boost by up to 20%
            boost = min(term_matches / len(query_terms) * 0.2, 0.2)
            adjusted_score = score * (1 - boost)  # Lower score is better in FAISS L2

            reranked_results.append((doc, adjusted_score))

        # Sort by adjusted score (ascending for L2 distance)
        reranked_results.sort(key=lambda x: x[1])

        return reranked_results

    # TODO: Implement keyword_search method for hybrid search
    # This would implement traditional keyword search (BM25, TF-IDF, etc.)

    # TODO: Implement filter_by_quarter method
    # This would filter results by specific quarters or fiscal years

    def get_metadata_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available metadata for filtering.

        Returns:
            Dictionary containing available quarters, years, etc.
        """
        return self.vector_store.get_stats()


def main():
    """
    Example usage of the RetrievalSystem class.
    This function demonstrates how to use the RetrievalSystem to retrieve
    relevant documents based on user queries.
    """
    # Import required classes for the example
    from pathlib import Path
    from amk.processing.embedding_generator import EmbeddingGenerator
    from amk.storage.faiss_storage import VectorStore

    # Set up paths
    resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
    index_path = resources_path / "faiss"

    # 1. Load the vector store
    vector_store = VectorStore.load_index(index_path)
    print(f"Loaded vector store with {len(vector_store.documents)} documents")

    # 2. Initialize embedding generator
    embedding_generator = EmbeddingGenerator()

    # 3. Initialize the retrieval system
    retrieval_system = RetrievalController(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        top_k=3
    )

    # 4. Test retrieval with a sample query
    query = "What were Salesforce's financial results in the most recent quarter?"
    print(f"\nRetrieving documents for query: '{query}'")

    # 5. Retrieve relevant documents
    results = retrieval_system.retrieve(query)

    print(f"Retrieved {len(results)} documents")

    # 6. Display results
    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}:")
        source = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))
        print(f"  Source: {source}")
        print(f"  Content (first 150 chars): {doc.page_content[:150]}...")

    # Log completion
    print("\nRetrieval system demo complete!")


if __name__ == "__main__":
    main()