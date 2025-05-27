"""
DocumentProcessor module for processing and chunking documents for a RAG system.
This module processes documents loaded by DocumentIngestion, breaking them into
appropriately sized chunks for embedding and retrieval while preserving context
and important metadata.
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv
# For chunking and text splitting
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

# For LLM-based metadata extraction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()

class DocumentProcessor:
    """
    A class for processing documents and splitting them into chunks suitable for
    embedding and retrieval in a RAG system.

    This class handles:
    - Cleaning and normalizing document text
    - Splitting documents into chunks with appropriate overlap
    - Preserving metadata and context across chunks
    - Supporting different chunking strategies
    - LLM-based metadata extraction
    """

    def __init__(
            self,
            chunk_size: int = 5000,
            chunk_overlap: int = 400
    ):
        """
        Initialize the DocumentProcessor with chunking parameters.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Always use token-based splitting (better for LLM context windows)
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size // 4,  # Approximate tokens from characters
            chunk_overlap=chunk_overlap // 4
        )

        # TODO: Extend chunking strategies in the future
        # Consider adding: semantic chunking, recursive character splitting,
        # or domain-specific chunking strategies for financial documents

        openai.api_key = os.environ.get("OPENAI_API_KEY")

        # Initialize LLM for metadata extraction
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=1000
        )
        logger.info("Initialized ChatOpenAI for metadata extraction")

        # Initialize JSON output parser
        self.json_parser = JsonOutputParser()

        # Define prompt template for metadata extraction
        self.metadata_prompt = ChatPromptTemplate.from_template("""
You are an expert at analyzing financial earnings call transcripts. 
Extract structured metadata from the following document content.

Document content (first 2000 characters):
{content}

Extract the following information and return as valid JSON:
{{
    "quarter": "Q1, Q2, Q3, or Q4 if mentioned, otherwise null",
    "fiscal_year": "Year in YYYY format if mentioned, otherwise null",
    "call_date": "Date of earnings call if mentioned, otherwise null",
    "company": "Company name if mentioned, otherwise null",
    "key_speakers": ["List of key speakers mentioned (CEO, CFO, etc.), max 5"],
    "main_topics": ["List of main topics discussed, max 5"],
    "sample_questions": ["List of 3-5 questions users might ask about this content"],
    "document_summary": "Brief 2-3 sentence summary of the document content"
}}

Only include information that is explicitly mentioned in the content. 
Use null for missing information.
Ensure the output is valid JSON.
""")

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Original document text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', text)

        # Normalize line endings
        cleaned_text = cleaned_text.replace('\r\n', '\n')

        # TODO: Add more sophisticated cleaning for financial transcripts
        # For example, standardizing speaker notations or formatting

        return cleaned_text

    def _extract_metadata_with_llm(self, document: Document) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from document content using LLM.
        Uses disk cache to avoid re-processing the same content.

        Args:
            document: Langchain Document object

        Returns:
            Dictionary containing extracted metadata
        """
        import hashlib
        import pickle
        from pathlib import Path

        try:
            resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
            cache_dir = resources_path / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Use first 2000 characters for metadata extraction to stay within token limits
            content_preview = document.page_content[:2000]

            # Create cache key from content hash
            cache_file = cache_dir / f"{document.metadata.get('filename')}.pkl"

            # Check if metadata is already cached
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_metadata = pickle.load(f)
                    logger.info(
                        f"Retrieved cached metadata for document: {document.metadata.get('filename', 'unknown')}")
                    return cached_metadata
                except Exception as cache_error:
                    logger.warning(f"Failed to load cached metadata: {cache_error}")
                    # Continue to generate new metadata

            if not self.llm:
                logger.error("LLM not available for metadata extraction")
                # TODO: Future enhancement - implement rule-based fallback metadata extraction
                # This could include regex patterns for quarter info, speaker detection,
                # basic topic extraction, and simple question generation
                return {
                    "quarter": None,
                    "fiscal_year": None,
                    "call_date": None,
                    "company": None,
                    "key_speakers": [],
                    "main_topics": [],
                    "sample_questions": [],
                    "document_summary": None
                }

            # Create the chain
            chain = self.metadata_prompt | self.llm | self.json_parser

            # Extract metadata
            extracted_metadata = chain.invoke({"content": content_preview})

            # Cache the extracted metadata
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(extracted_metadata, f)
                logger.info(f"Cached metadata for future use: {cache_file}")
            except Exception as cache_error:
                logger.warning(f"Failed to cache metadata: {cache_error}")

            logger.info(
                f"Successfully extracted metadata using LLM for document: {document.metadata.get('filename', 'unknown')}")

            return extracted_metadata

        except Exception as e:
            logger.error(f"Error extracting metadata with LLM: {e}")
            # TODO: Future enhancement - implement rule-based fallback metadata extraction
            return {
                "quarter": None,
                "fiscal_year": None,
                "call_date": None,
                "company": None,
                "key_speakers": [],
                "main_topics": [],
                "sample_questions": [],
                "document_summary": None
            }

    def process_document(self, document: Document) -> List[Document]:
        """
        Process a single document and split it into chunks.

        Args:
            document: Langchain Document object to process

        Returns:
            List of chunked Document objects
        """
        logger.info(f"Processing document: {document.metadata.get('filename', 'unknown')}")

        # Clean text content
        cleaned_text = self._clean_text(document.page_content)

        # Extract additional metadata using LLM
        extracted_metadata = self._extract_metadata_with_llm(document)

        # Create a document with cleaned text and enhanced metadata
        enhanced_metadata = {**document.metadata, **extracted_metadata}
        cleaned_document = Document(page_content=cleaned_text, metadata=enhanced_metadata)

        # Split the document into chunks
        chunks = self.text_splitter.split_documents([cleaned_document])

        # Add chunk index to metadata and preserve extracted metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

            # TODO: Future optimization - only extract metadata once per document
            # and share across all chunks, rather than duplicating for each chunk

        logger.info(f"Split document into {len(chunks)} chunks")

        return chunks

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of documents and split them into chunks.

        Args:
            documents: List of Langchain Document objects

        Returns:
            List of chunked Document objects
        """
        all_chunks = []

        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)

        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} total chunks")

        return all_chunks


def main():
    """
    Example usage of the DocumentProcessor class.
    This function demonstrates how to process documents loaded by DocumentIngestion.
    """
    # Import DocumentIngestion for the example
    from document_ingestion import DocumentIngestion

    # Initialize the document ingestion class
    ingestion = DocumentIngestion()

    # Load documents from a directory
    documents = ingestion.load_directory()

    print(f"Loaded {len(documents)} document segments")

    # Initialize the document processor
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Process the documents into chunks
    chunks = processor.process_documents(documents)

    # Print summary of processed chunks
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # Print sample of the first chunk if available
    if chunks:
        print("\nSample chunk metadata:")
        for key, value in chunks[0].metadata.items():
            print(f"  {key}: {value}")

        print("\nSample chunk content (first 200 chars):")
        print(chunks[0].page_content[:200] + "...")

    # Log completion
    print("\nDocument processing complete!")


if __name__ == "__main__":
    main()