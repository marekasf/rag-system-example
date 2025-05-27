"""
VectorStore module for storing and retrieving vector embeddings in a RAG system.
This module provides functionality to store document embeddings in a vector database,
perform similarity searches, and support metadata filtering for efficient retrieval.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# FAISS for vector storage
import faiss
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorStore:
    """
    A class for storing and retrieving vector embeddings using FAISS.

    This class handles:
    - Storing document embeddings in a FAISS index
    - Performing similarity searches
    - Supporting metadata filtering
    - Allowing for index updates as new documents are added
    """

    def __init__(self, dimension: int = 384, index_type: str = "L2"):
        """
        Initialize the VectorStore with specified parameters.

        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index to use ("L2" for L2 distance, "IP" for inner product)
        """
        self.dimension = dimension
        self.index_type = index_type

        # Initialize FAISS index based on type
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            logger.warning(f"Unsupported index type: {index_type}, defaulting to L2")
            self.index = faiss.IndexFlatL2(dimension)

        # Storage for document content and metadata
        self.documents = []
        self.metadata_list = []

        logger.info(f"Initialized FAISS vector store with dimension {dimension} and index type {index_type}")

    def add_documents(self, embedded_documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.

        Args:
            embedded_documents: List of dictionaries containing document content, metadata, and embedding
        """
        if not embedded_documents:
            logger.warning("No documents provided to add to vector store")
            return

        # Extract embeddings, content, and metadata
        embeddings = [doc["embedding"] for doc in embedded_documents]

        # Convert embeddings list to numpy array for FAISS
        embeddings_array = np.array(embeddings).astype('float32')

        # Get the current index size
        current_size = len(self.documents)

        # Store document content and metadata
        for doc in embedded_documents:
            self.documents.append(doc["content"])
            self.metadata_list.append(doc["metadata"])

        # Add embeddings to FAISS index
        self.index.add(embeddings_array)

        logger.info(f"Added {len(embedded_documents)} documents to vector store. Total: {len(self.documents)}")

    def similarity_search(
            self,
            query_embedding: List[float],
            k: int = 4,
            filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search to find the most similar documents to a query.

        Args:
            query_embedding: Embedding vector for the query
            k: Number of results to return
            filter_metadata: Optional dictionary for filtering results by metadata

        Returns:
            List of tuples containing Document objects and similarity scores
        """
        if not self.documents:
            logger.warning("No documents in vector store to search")
            return []

        # Convert query_embedding to numpy array
        query_array = np.array([query_embedding]).astype('float32')

        # Adjust k if there are fewer documents than requested
        k = min(k, len(self.documents))

        # Perform similarity search
        # D contains distances, I contains indices
        D, I = self.index.search(query_array, k * 4)  # Retrieve more to allow for filtering

        results = []

        # Process search results
        for i, idx in enumerate(I[0]):
            # Skip invalid indices
            if idx == -1 or idx >= len(self.documents):
                continue

            # Get document content and metadata
            content = self.documents[idx]
            metadata = self.metadata_list[idx]
            distance = D[0][i]

            # Apply metadata filtering if specified
            if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                continue

            # Create Document object
            doc = Document(page_content=content, metadata=metadata)

            # Add to results
            results.append((doc, distance))

            # Stop if we have enough results
            if len(results) >= k:
                break

        # If we don't have enough results after filtering, add more
        if len(results) < k and len(results) < len(self.documents):
            logger.info(f"Only found {len(results)} results after filtering, need {k}")
            # TODO: Implement more sophisticated handling for when filtering reduces results below k

        return results

    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """
        Check if document metadata matches filter criteria.

        Args:
            metadata: Document metadata
            filter_metadata: Filter criteria

        Returns:
            True if document matches all filter criteria, False otherwise
        """
        for key, value in filter_metadata.items():
            # Skip if key is not in metadata
            if key not in metadata:
                return False

            # Check if value matches
            if metadata[key] != value:
                return False

        return True

    def save_index(self, directory: str | Path, filename: str = "vector_store") -> None:
        """
        Save the vector store index and associated data to disk.

        Args:
            directory: Directory to save the index to
            filename: Base filename for saved files
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(directory, f"{filename}.index")
        faiss.write_index(self.index, index_path)

        # Save documents and metadata
        data_path = os.path.join(directory, f"{filename}.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump((self.documents, self.metadata_list, self.dimension, self.index_type), f)

        logger.info(f"Saved vector store to {directory}")

    @classmethod
    def load_index(cls, directory: str | Path, filename: str = "vector_store") -> 'VectorStore':
        """
        Load a vector store index and associated data from disk.

        Args:
            directory: Directory to load the index from
            filename: Base filename for saved files

        Returns:
            VectorStore instance with loaded data
        """
        # Load FAISS index
        index_path = os.path.join(directory, f"{filename}.index")
        index = faiss.read_index(index_path)

        # Load documents and metadata
        data_path = os.path.join(directory, f"{filename}.pkl")
        with open(data_path, 'rb') as f:
            documents, metadata_list, dimension, index_type = pickle.load(f)

        # Create new VectorStore instance
        vector_store = cls(dimension=dimension, index_type=index_type)
        vector_store.index = index
        vector_store.documents = documents
        vector_store.metadata_list = metadata_list

        logger.info(f"Loaded vector store from {directory} with {len(documents)} documents")

        return vector_store

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID from metadata.

        Args:
            doc_id: Document ID to search for

        Returns:
            Document if found, None otherwise
        """
        # TODO: Implement more efficient lookup by maintaining an ID-to-index mapping

        for i, metadata in enumerate(self.metadata_list):
            if metadata.get("doc_id") == doc_id:
                return Document(page_content=self.documents[i], metadata=metadata)

        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing vector store statistics
        """
        # Get unique document sources
        sources = set()
        quarters = set()
        fiscal_years = set()

        for metadata in self.metadata_list:
            source = metadata.get("source")
            if source:
                sources.add(source)

            quarter = metadata.get("quarter")
            if quarter:
                quarters.add(quarter)

            fiscal_year = metadata.get("fiscal_year")
            if fiscal_year:
                fiscal_years.add(fiscal_year)

        stats = {
            "total_documents": len(self.documents),
            "unique_sources": len(sources),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "unique_quarters": list(quarters) if quarters else [],
            "fiscal_years": list(fiscal_years) if fiscal_years else []
        }

        return stats


def main():
    """
    Example usage of the VectorStore class.
    This function demonstrates how to create a vector store, add documents,
    and perform similarity searches.
    """
    # Import required classes for the example
    from amk.processing.document_ingestion import DocumentIngestion
    from amk.processing.document_procesor import DocumentProcessor
    from amk.processing.embedding_generator import EmbeddingGenerator

    # 1. Load documents using DocumentIngestion
    ingestion = DocumentIngestion()
    documents = ingestion.load_directory()
    print(f"Loaded {len(documents)} document segments")

    # 2. Process documents into chunks using DocumentProcessor
    processor = DocumentProcessor()
    chunks = processor.process_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # 3. Generate embeddings using EmbeddingGenerator
    # For demo, just use a few chunks to avoid excessive API calls
    demo_chunks = chunks[:5] if len(chunks) > 5 else chunks

    embedding_generator = EmbeddingGenerator()
    embedded_documents = embedding_generator.generate_embeddings(demo_chunks)
    print(f"Generated embeddings for {len(embedded_documents)} document chunks")

    # 4. Create vector store and add documents
    vector_store = VectorStore(
        dimension=len(embedded_documents[0]["embedding"]) if embedded_documents else 384
    )

    vector_store.add_documents(embedded_documents)
    print(f"Added {len(embedded_documents)} documents to vector store")

    # 5. Test similarity search
    query = "What were Salesforce's financial results in the Q3 2023?"
    query_embedding = embedding_generator.embed_query(query)

    results = vector_store.similarity_search(query_embedding, k=2)
    print(f"\nSearch results for query: '{query}'")

    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i + 1} (distance: {score:.4f}):")
        print(f"Metadata: {doc.metadata}")
        print(f"Content (first 200 chars): {doc.page_content[:200]}...")

    # 6. Test metadata filtering
    if len(vector_store.metadata_list) > 0:
        sample_metadata = vector_store.metadata_list[0]
        filter_key = next(iter(sample_metadata.keys()))
        filter_value = sample_metadata[filter_key]

        filter_metadata = {filter_key: filter_value}
        filtered_results = vector_store.similarity_search(query_embedding, k=2, filter_metadata=filter_metadata)

        print(f"\nFiltered search results with {filter_key}={filter_value}:")
        for i, (doc, score) in enumerate(filtered_results):
            print(f"\nFiltered Result {i + 1} (distance: {score:.4f}):")
            print(f"Metadata: {doc.metadata}")
            print(f"Content (first 100 chars): {doc.page_content[:100]}...")

    # 7. Save and load index (for demo purposes)
    resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
    temp_dir = resources_path / "faiss" / "temp_index"
    vector_store.save_index(temp_dir)

    # Load the saved index
    loaded_store = VectorStore.load_index(temp_dir)
    print(f"\nLoaded vector store with {len(loaded_store.documents)} documents")

    # Get statistics
    stats = vector_store.get_stats()
    print("\nVector store statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Log completion
    print("\nVector store demo complete!")


if __name__ == "__main__":
    main()