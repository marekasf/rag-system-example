"""
EmbeddingGenerator module for generating vector embeddings from document chunks.
This module converts processed document chunks into vector embeddings suitable
for semantic search in a RAG system, with focus on financial domain content.
"""

import logging
import os
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class EmbeddingGenerator:
    """
    A class for generating vector embeddings from document chunks for a RAG system.

    This class handles:
    - Converting text chunks to vector embeddings
    - Supporting different embedding models
    - Batching for efficiency
    - Caching to avoid redundant embedding generation
    """

    def __init__(
            self,
            embedding_model: str = "huggingface",
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            batch_size: int = 16,
    ):
        """
        Initialize the EmbeddingGenerator with specified parameters.

        This class supports both OpenAI and HuggingFace embedding models.:
        embedding_model="openai",  # Alternative: "huggingface"
        model_name="text-embedding-3-large",  # For HF: "sentence-transformers/all-MiniLM-L6-v2"

        Args:
            embedding_model: Type of embedding model to use ("openai" or "huggingface")
            model_name: Specific model name for the chosen embedding type
            batch_size: Number of documents to process at once
        """
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.batch_size = batch_size

        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            logger.info("OpenAI API key loaded from environment variable.")
            openai.api_key = self.openai_api_key

        # Initialize embedding model based on type
        if embedding_model.lower() == "huggingface":
            # Use HuggingFace embeddings as alternative
            self.embedder = HuggingFaceEmbeddings(
                model_name=model_name
            )
            logger.info(f"Initialized HuggingFace embedding model: {model_name}")
        else:
            # Use OpenAI embeddings
            if not self.openai_api_key:
                logger.warning(
                    "No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass key to constructor.")

            self.embedder = OpenAIEmbeddings(model=model_name)
            logger.info(f"Initialized OpenAI embedding model: {model_name}")

    def generate_embeddings(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of document chunks.

        Args:
            documents: List of Langchain Document objects to embed

        Returns:
            List of dictionaries containing document content, metadata, and embedding
        """
        if not documents:
            logger.warning("No documents provided for embedding generation.")
            return []

        logger.info(f"Generating embeddings for {len(documents)} documents using {self.embedding_model} model.")

        # Extract text content from documents and append **all** metadata
        texts = []
        for doc in documents:
            text = doc.page_content

            # Append all metadata key-value pairs to the text
            if doc.metadata:
                metadata_str = " ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                text += " " + metadata_str

            texts.append(text)

        # Process in batches for efficiency
        embeddings_list = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_docs = documents[i:i + self.batch_size]

            try:
                # Generate embeddings for the batch
                batch_embeddings = self.embedder.embed_documents(batch_texts)

                # Combine documents with their embeddings
                for j, embedding in enumerate(batch_embeddings):
                    doc = batch_docs[j]
                    embeddings_list.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": embedding
                    })

                logger.info(
                    f"Generated embeddings for batch {i // self.batch_size + 1}/{(len(texts) - 1) // self.batch_size + 1}")

            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {str(e)}")
                continue

        logger.info(f"Successfully generated {len(embeddings_list)} embeddings")
        return embeddings_list


    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query string

        Returns:
            Embedding vector for the query
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedder.embed_query(query)
            return query_embedding

        except Exception as e:
            logger.error(f"Error generating embedding for query: {str(e)}")
            # Return empty embedding in case of error
            # In a production system, this should be handled more gracefully
            return []


def main():
    """
    Example usage of the EmbeddingGenerator class.
    This function demonstrates how to generate embeddings for documents
    processed by DocumentIngestion and DocumentProcessor.
    """
    # Import required classes for the example
    from amk.processing.document_ingestion import DocumentIngestion
    from amk.processing.document_procesor import DocumentProcessor

    # 1. Load documents using DocumentIngestion
    ingestion = DocumentIngestion()
    documents = ingestion.load_directory()
    print(f"Loaded {len(documents)} document segments")

    # 2. Process documents into chunks using DocumentProcessor
    processor = DocumentProcessor()
    chunks = processor.process_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # 3. Generate embeddings using EmbeddingGenerator
    # Note: For demo purposes, using a smaller batch to avoid API rate limits
    embedding_generator = EmbeddingGenerator()

    # For demo, just use first few chunks to avoid excessive API calls
    demo_chunks = chunks[:5] if len(chunks) > 5 else chunks

    # Generate embeddings for the chunks
    embedded_documents = embedding_generator.generate_embeddings(demo_chunks)

    # Print summary and sample
    print(f"\nGenerated embeddings for {len(embedded_documents)} document chunks")

    if embedded_documents:
        print("\nSample embedding metadata:")
        for key, value in embedded_documents[0]["metadata"].items():
            print(f"  {key}: {value}")

        print("\nEmbedding vector dimensions:")
        print(f"  {len(embedded_documents[0]['embedding'])} dimensions")

        # Print the first few values of the embedding vector
        print("\nSample embedding vector (first 5 values):")
        print(embedded_documents[0]["embedding"][:5])

        # Example of query embedding
        query = "What were Salesforce's financial results in the most recent quarter?"
        query_embedding = embedding_generator.embed_query(query)

        print(f"\nGenerated query embedding for: '{query}'")
        print(f"Query embedding dimensions: {len(query_embedding)}")

    # Log completion
    print("\nEmbedding generation complete!")


if __name__ == "__main__":
    main()