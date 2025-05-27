"""
DocumentIngestion module for loading and processing PDF/text documents for a RAG system.
This module provides functionality to load documents from specified directories,
extract their content, and prepare them for embedding and retrieval.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

# Third-party imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIngestion:
    """
    A class for ingesting documents from various sources into a standardized format
    suitable for RAG (Retrieval Augmented Generation) systems.

    This class handles:
    - Loading documents from a file or directory
    - Extracting content while preserving structural information
    - Extracting metadata (date, quarter, year, etc.)
    - Configurable PDF loading modes (page vs single document)
    """

    def __init__(self, pdf_mode: str = "single", pages_delimiter: str = "\n\n--- PAGE BREAK ---\n\n"):
        """
        Initialize the DocumentIngestion class with default settings.

        Args:
            pdf_mode: Mode for PDF loading ("page" or "single")
            pages_delimiter: Delimiter to use when pdf_mode="single" to mark page boundaries
        """
        self.pdf_mode = pdf_mode
        self.pages_delimiter = pages_delimiter

        # Define supported file extensions
        self.supported_extensions = {
            '.pdf': self._create_pdf_loader,
            '.txt': TextLoader
        }

    def _create_pdf_loader(self, file_path: str):
        """
        Create a PyPDFLoader with the specified mode.

        Args:
            file_path: Path to the PDF file

        Returns:
            Configured PyPDFLoader instance
        """
        if self.pdf_mode == "single":
            return PyPDFLoader(
                file_path,
                mode="single",
                pages_delimiter=self.pages_delimiter
            )
        else:
            return PyPDFLoader(file_path, mode="page")

    def _extract_metadata(self, file_path: str, context = None) -> Dict[str, Any]:
        """
        Extract metadata from file path and potentially from content.

        Args:
            file_path: Path to the document file
            context: Content of the document (if applicable)

        Returns:
            Dictionary containing metadata (filename, date, etc.)
        """
        # Basic metadata from file
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        file_stats = os.stat(file_path)

        # Initialize with basic metadata
        metadata = {
            "source": file_path,
            "filename": file_name,
            "file_extension": file_ext,
            "file_size": file_stats.st_size,
            "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "pdf_mode": self.pdf_mode  # Add PDF mode to metadata
        }

        # TODO: Add more sophisticated metadata extraction from content
        # For example, extracting quarter, year, and other relevant information
        # This would be specific to the format of the documents being processed

        return metadata

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single document file and convert it to Langchain Document objects.

        Args:
            file_path: Path to the document file

        Returns:
            List of Langchain Document objects
        """
        logger.info(f"Loading file: {file_path} with PDF mode: {self.pdf_mode}")

        file_ext = os.path.splitext(file_path)[1].lower()

        # Check if file extension is supported
        if file_ext not in self.supported_extensions:
            logger.warning(f"Unsupported file extension: {file_ext}")
            return []

        documents = []

        # Use LangChain document loaders
        loader_factory = self.supported_extensions[file_ext]

        # Handle PDF vs other file types
        if file_ext == '.pdf':
            loader = loader_factory(file_path)  # Uses _create_pdf_loader
        else:
            loader = loader_factory(file_path)  # Direct instantiation for TextLoader

        try:
            # Load documents using LangChain
            docs = loader.load()

            # Enhance metadata
            for doc in docs:
                # Extract and add additional metadata
                additional_metadata = self._extract_metadata(file_path, doc.page_content)

                # Merge additional metadata with existing metadata
                if hasattr(doc, 'metadata'):
                    doc.metadata.update(additional_metadata)
                else:
                    doc.metadata = additional_metadata

            documents.extend(docs)
            logger.info(f"Successfully loaded {len(docs)} document segments from {file_path}")

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")

        return documents

    def load_directory(self, directory_path: str | Path = None) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to directory containing documents
        Returns:
            List of Langchain Document objects
        """
        if not directory_path:
            resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
            directory_path = resources_path / "data"

        logger.info(f"Loading documents from directory: {directory_path} with PDF mode: {self.pdf_mode}")

        all_documents = []

        # Handle PDF files separately due to custom loader configuration
        pdf_files = list(Path(directory_path).rglob("*.pdf"))
        for pdf_file in pdf_files:
            docs = self.load_file(str(pdf_file))
            all_documents.extend(docs)

        # Handle other file types using DirectoryLoader
        for ext, loader_factory in self.supported_extensions.items():
            if ext == '.pdf':
                continue  # Already handled above

            # Create a pattern to match this extension
            glob_pattern = f"**/*{ext}"

            # Initialize directory loader
            try:
                directory_loader = DirectoryLoader(
                    directory_path,
                    glob=glob_pattern,
                    loader_cls=loader_factory,  # For non-PDF, this is just the class
                    show_progress=True
                )

                # Load documents
                docs = directory_loader.load()
                logger.info(f"Loaded {len(docs)} documents with extension {ext}")

                # Enhance metadata for each document
                for doc in docs:
                    additional_metadata = self._extract_metadata(doc.metadata.get('source', ''))
                    doc.metadata.update(additional_metadata)

                all_documents.extend(docs)

            except Exception as e:
                logger.error(f"Error loading {ext} files from {directory_path}: {str(e)}")

        logger.info(f"Total loaded documents: {len(all_documents)}")
        return all_documents


def main():
    """
    Example usage of the DocumentIngestion class with different PDF modes.
    This function demonstrates how to load documents from a directory.
    """

    print("=== PDF Mode: page (default behavior) ===")
    # Initialize with page mode (splits each page as separate document)
    ingestion_page = DocumentIngestion(pdf_mode="page")
    documents_page = ingestion_page.load_directory()
    print(f"Page mode: Loaded {len(documents_page)} document segments")

    print("\n=== PDF Mode: single (combine all pages) ===")
    # Initialize with single mode (entire PDF as one document)
    ingestion_single = DocumentIngestion(pdf_mode="single", pages_delimiter="\n\n--- PAGE BREAK ---\n\n")
    documents_single = ingestion_single.load_directory()
    print(f"Single mode: Loaded {len(documents_single)} document segments")

    # Print sample of the first document if available
    if documents_single:
        print("\nSample document metadata (single mode):")
        for key, value in documents_single[0].metadata.items():
            print(f"  {key}: {value}")

        print("\nSample document content (first 200 chars):")
        print(documents_single[0].page_content[:200] + "...")

        # Show where page breaks appear
        content = documents_single[0].page_content
        if "PAGE BREAK" in content:
            first_break = content.find("--- PAGE BREAK ---")
            if first_break != -1:
                print(f"\nPage break found at position {first_break}")
                print("Content around first page break:")
                print(content[max(0, first_break-50):first_break+100])

    # Log completion
    print("\nDocument ingestion complete!")


if __name__ == "__main__":
    main()