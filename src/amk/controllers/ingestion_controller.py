"""
IngestController module to coordinate the end-to-end process of:
1. Ingesting documents from a directory
2. Processing documents into chunks
3. Generating embeddings for chunks
4. Creating and saving a vector index

This module provides a high-level interface to orchestrate the entire document
ingestion pipeline for a RAG system.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

# Import components from our RAG system
from amk.processing.document_ingestion import DocumentIngestion
from amk.processing.document_procesor import DocumentProcessor
from amk.processing.embedding_generator import EmbeddingGenerator
from amk.storage.faiss_storage import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestController:
    """
    A controller class that orchestrates the entire document ingestion process
    for a RAG system.

    This class coordinates:
    - Document loading from files
    - Document processing and chunking
    - Embedding generation
    - Vector index creation and storage
    """

    def __init__(
            self,
            embedding_model: str = "huggingface",
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            chunk_size: int = 5000,
            chunk_overlap: int = 400,
            batch_size: int = 16
    ):
        """
        Initialize the IngestController with configuration for all pipeline components.

        Args:
            embedding_model: Type of embedding model to use ("openai" or "huggingface")
            model_name: Specific model name for the chosen embedding type
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            batch_size: Batch size for embedding generation
        """
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        # Initialize the pipeline components
        self.document_ingestion = DocumentIngestion()
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator(
            embedding_model=embedding_model,
            model_name=model_name,
            batch_size=batch_size
        )

        # Will be set after embedding generation
        self.vector_store = None

        logger.info("IngestController initialized with embedding model: " +
                    f"{embedding_model}/{model_name}, chunk size: {chunk_size}, " +
                    f"overlap: {chunk_overlap}")

    def ingest_directory(
            self,
            input_directory: Optional[Path] = None,
            output_directory: Optional[Path] = None,
            save_intermediate: bool = False
    ) -> VectorStore:
        """
        Process all documents in a directory through the complete ingestion pipeline.

        Args:
            input_directory: Directory containing documents to ingest
            output_directory: Directory to save the vector index
            save_intermediate: Whether to save intermediate results (chunks, embeddings)

        Returns:
            Initialized VectorStore instance
        """
        start_time = time.time()

        # Set default paths if not provided
        if input_directory is None:
            resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
            input_directory = resources_path / "data"

        if output_directory is None:
            resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
            output_directory = resources_path / "faiss"

        # Ensure directories exist
        os.makedirs(input_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

        logger.info(f"Starting ingestion from {input_directory}")
        logger.info(f"Output will be saved to {output_directory}")

        # Step 1: Load documents
        logger.info("Step 1: Loading documents")
        documents = self.document_ingestion.load_directory(input_directory)
        logger.info(f"Loaded {len(documents)} document segments")

        if not documents:
            logger.warning("No documents found in the specified directory. Ingestion aborted.")
            raise ValueError("No documents found in the specified directory.")

        # Step 2: Process documents into chunks
        logger.info("Step 2: Processing documents into chunks")
        chunks = self.document_processor.process_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Save intermediate chunks if requested
        if save_intermediate:
            self._save_chunks(chunks, output_directory)

        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings")
        embedded_documents = self.embedding_generator.generate_embeddings(chunks)
        logger.info(f"Generated {len(embedded_documents)} embeddings")

        # Save intermediate embeddings if requested
        if save_intermediate:
            # TODO: Implement saving embeddings as intermediate results
            pass

        # Step 4: Create vector store
        logger.info("Step 4: Creating vector store")
        # Determine embedding dimension from the first embedded document
        if embedded_documents:
            embedding_dim = len(embedded_documents[0]["embedding"])
        else:
            # Default dimension for the chosen model
            embedding_dim = 384  # Default for MiniLM-L6-v2
            logger.warning(f"No embeddings generated, using default dimension: {embedding_dim}")

        self.vector_store = VectorStore(dimension=embedding_dim)
        self.vector_store.add_documents(embedded_documents)
        logger.info(f"Added {len(embedded_documents)} documents to vector store")

        # Step 5: Save vector store
        logger.info("Step 5: Saving vector store")
        self.vector_store.save_index(output_directory)
        logger.info(f"Saved vector store to {output_directory}")

        # Get and log statistics
        stats = self.vector_store.get_stats()
        logger.info("Vector store statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        elapsed_time = time.time() - start_time
        logger.info(f"Ingestion completed in {elapsed_time:.2f} seconds")

        return self.vector_store

    def _save_chunks(self, chunks: List, output_directory: Path) -> None:
        """
        Save document chunks as intermediate results.

        Args:
            chunks: List of document chunks
            output_directory: Directory to save the chunks
        """
        import pickle

        chunks_path = output_directory / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(chunks, f)

        logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")


def main():
    """
    Example usage of the IngestController class.
    This function demonstrates how to ingest documents from a directory
    and create a vector index.
    """
    # Set up paths
    resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
    input_path = resources_path / "data"
    output_path = resources_path / "faiss"

    # Create IngestController with default settings
    controller = IngestController()

    # Run ingestion process
    print(f"Starting ingestion from {input_path}")
    print(f"Output will be saved to {output_path}")

    vector_store = controller.ingest_directory(
        input_directory=input_path,
        output_directory=output_path,
        save_intermediate=True
    )

    # Test the vector store with a sample query
    if vector_store:
        print("\nTesting vector store with a sample query")
        query = "What were Salesforce's financial results in the most recent quarter?"
        query_embedding = controller.embedding_generator.embed_query(query)

        results = vector_store.similarity_search(query_embedding, k=2)

        print(f"\nSearch results for query: '{query}'")
        for i, (doc, score) in enumerate(results):
            print(f"\nResult {i + 1} (distance: {score:.4f}):")
            print(f"Metadata: {doc.metadata}")
            print(f"Content (first 150 chars): {doc.page_content[:150]}...")

    print("\nIngestion process complete!")


if __name__ == "__main__":
    main()