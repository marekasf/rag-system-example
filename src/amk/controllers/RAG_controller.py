"""
RAGController module for integrating retrieval and response generation in a RAG system.
This module combines the retrieval component with an LLM to generate responses for user
queries based on retrieved document context. This is the core component that produces
final responses in the conversational RAG system.
"""

import logging
import os
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Import our retrieval component
from amk.controllers.retrieval_controller import RetrievalController
from amk.processing.embedding_generator import EmbeddingGenerator
from amk.storage.faiss_storage import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class RAGController:
    """
    A class for coordinating retrieval and response generation in a RAG system.

    This class handles:
    - Retrieving relevant documents for user queries
    - Formatting prompts with retrieved context
    - Generating responses using an LLM
    - (Future) Maintaining conversation context
    - (Future) Generating citations
    """

    def __init__(
            self,
            retrieval_controller: RetrievalController,
            llm_model: str = "gpt-4",
            temperature: float = 0.1,
            max_tokens: int = 1000,
            top_k: int = 5
    ):
        """
        Initialize the RAGController with specified components.

        Args:
            retrieval_controller: RetrievalController for document retrieval
            llm_model: LLM model to use for response generation
            temperature: Temperature parameter for LLM (lower = more deterministic)
            max_tokens: Maximum number of tokens in generated responses
            top_k: Number of documents to retrieve
        """
        self.retrieval_controller = retrieval_controller
        self.top_k = top_k

        # Check for OpenAI API key in environment
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Initialize output parser
        self.output_parser = StrOutputParser()

        # Initialize conversation history (simple list for POC)
        # TODO: Implement more sophisticated conversation memory
        self.conversation_history = []

        logger.info(f"Initialized RAGController with LLM model {llm_model}, temperature {temperature}")

    def _format_retrieved_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a string for inclusion in the prompt.

        Args:
            docs: List of retrieved Document objects

        Returns:
            Formatted string containing content from retrieved documents
        """
        formatted_docs = ""

        for i, doc in enumerate(docs):
            # Extract source information
            source = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))

            # Extract quarter/date information if available
            quarter = doc.metadata.get('quarter', 'Unknown Quarter')
            fiscal_year = doc.metadata.get('fiscal_year', 'Unknown Year')

            # Add document with metadata
            formatted_docs += f"\n--- Document {i + 1} (Source: {source}, {quarter} {fiscal_year}) ---\n"
            formatted_docs += doc.page_content
            formatted_docs += "\n\n"

        return formatted_docs

    def _create_prompt_template(self, include_conversation_history: bool = False) -> ChatPromptTemplate:
        """
        Create a prompt template for the RAG system.

        Args:
            include_conversation_history: Whether to include conversation history

        Returns:
            ChatPromptTemplate for generating responses
        """
        # Basic template for financial Q&A without conversation history (POC)
        system_template = """You are a helpful financial analyst assistant focusing on Salesforce's quarterly earnings.
You answer questions based ONLY on the context provided below. 
If you don't know the answer or can't find it in the context, say "I don't have enough information to answer this question based on the available documents."

Be precise, factual, and focus on information from Salesforce's earnings calls.
Avoid making up information that isn't in the provided context.

Context information:
{context}

Answer the user's question based solely on the information in the context above.
"""

        # TODO: Add templates for different response types (Q&A vs. summarization)
        # TODO: Enhance prompt with instructions for citation

        if include_conversation_history:
            # TODO: Implement history-aware prompt template
            human_template = """Previous conversation:
{conversation_history}

Current question: {query}"""
        else:
            human_template = "{query}"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        return prompt_template

    def generate_response(
            self,
            query: str,
            filter_metadata: Optional[Dict[str, Any]] = None,
            include_sources: bool = False,
            include_conversation_history: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response to a user query using the RAG approach.

        Args:
            query: User query string
            filter_metadata: Optional metadata for filtering retrieval results
            include_sources: Whether to include source information in the response
            include_conversation_history: Whether to include conversation history

        Returns:
            Dictionary containing the generated response and metadata
        """
        logger.info(f"Generating response for query: '{query}'")

        # 1. Retrieve relevant documents
        retrieved_docs = self.retrieval_controller.retrieve(
            query=query,
            top_k=self.top_k,
            filter_metadata=filter_metadata
        )

        # Handle case with no retrieved documents
        if not retrieved_docs:
            logger.warning("No documents retrieved for query")
            return {
                "response": "I don't have enough information to answer this question based on the available documents.",
                "sources": [],
                "success": False
            }

        # 2. Format retrieved documents
        context = self._format_retrieved_docs(retrieved_docs)

        # 3. Create input dict for prompt
        input_dict = {
            "context": context,
            "query": query
        }

        # Include conversation history if requested
        if include_conversation_history and self.conversation_history:
            # Format conversation history
            # This is a simplified implementation for the POC
            history_text = "\n".join([
                f"Human: {turn['query']}\nAssistant: {turn['response']}"
                for turn in self.conversation_history
            ])
            input_dict["conversation_history"] = history_text

        # 4. Create prompt template
        prompt_template = self._create_prompt_template(include_conversation_history)

        # 5. Generate response using LLM
        try:
            # Chain: prompt -> LLM -> output parser
            chain = prompt_template | self.llm | self.output_parser
            response = chain.invoke(input_dict)
            success = True

            # TODO: Implement more sophisticated error handling
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response = "I'm sorry, there was an error generating a response. Please try again."
            success = False

        # 6. Update conversation history
        self.conversation_history.append({
            "query": query,
            "response": response
        })

        # Limit history length
        # TODO: Implement more sophisticated history management
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        # 7. Prepare result object
        result = {
            "response": response,
            "success": success
        }

        # Include sources if requested
        if include_sources:
            # TODO: Implement proper citation extraction
            sources = []
            for doc in retrieved_docs:
                source = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))
                quarter = doc.metadata.get('quarter', 'Unknown Quarter')
                fiscal_year = doc.metadata.get('fiscal_year', 'Unknown Year')

                sources.append({
                    "source": source,
                    "quarter": quarter,
                    "fiscal_year": fiscal_year
                })

            result["sources"] = sources

        return result

    def generate_summary(
            self,
            topic: str,
            filter_metadata: Optional[Dict[str, Any]] = None,
            max_length: int = 500
    ) -> Dict[str, Any]:
        """
        Generate a summary of a specific topic using the RAG approach.

        Args:
            topic: Topic to summarize
            filter_metadata: Optional metadata for filtering retrieval results
            max_length: Maximum length of the summary in characters

        Returns:
            Dictionary containing the generated summary and metadata
        """
        # TODO: Implement specialized summarization
        # For POC, we'll reuse the QA functionality with a summarization query

        summary_query = f"Summarize the information about {topic} from Salesforce's earnings calls"

        # Use the QA functionality with a specialized prompt
        result = self.generate_response(
            query=summary_query,
            filter_metadata=filter_metadata,
            include_sources=True
        )

        # Label it as a summary rather than a response
        if "response" in result:
            result["summary"] = result.pop("response")

        return result

    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.

        Returns:
            None
        """
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.

        Returns:
            Dictionary containing system statistics
        """
        # Get metadata from vector store via retrieval controller
        metadata_summary = self.retrieval_controller.get_metadata_summary()

        # Add conversation statistics
        metadata_summary["conversation_turns"] = len(self.conversation_history)
        metadata_summary["llm_model"] = self.llm.model_name

        return metadata_summary


def main():
    """
    Example usage of the RAGController class.
    This function demonstrates a simple happy path for using the RAG system.
    """
    from pathlib import Path

    # Check if OpenAI API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
        return

    # Set up paths
    resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
    index_path = resources_path / "faiss"

    print("Setting up RAG system...")

    # 1. Load vector store
    vector_store = VectorStore.load_index(index_path)
    print(f"Loaded vector store with {len(vector_store.documents)} documents")

    # 2. Initialize embedding generator
    embedding_generator = EmbeddingGenerator()

    # 3. Initialize retrieval controller
    retrieval_controller = RetrievalController(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        top_k=3
    )

    # 4. Initialize RAG controller
    rag_controller = RAGController(
        retrieval_controller=retrieval_controller,
        top_k=3
    )

    # 5. Get basic system statistics
    stats = rag_controller.get_system_stats()
    print(
        f"System has {stats.get('total_documents', 'Unknown')} documents across {stats.get('unique_sources', 'Unknown')} sources")

    # 6. Process a user query - simple happy path example
    user_query = "What were Salesforce's financial results in the Q3 2023?"
    print(f"\nUser query: '{user_query}'")

    # 7. Generate response with the RAG system
    print("Generating response...")
    result = rag_controller.generate_response(user_query, include_sources=True)

    # 8. Display response
    print("\nRAG System Response:")
    print(result["response"])

    # 9. Show sources (if available)
    if "sources" in result and result["sources"]:
        print("\nSources:")
        for i, source in enumerate(result["sources"][:3]):  # Show just the first 3 sources
            source_name = source.get('source', 'Unknown')
            quarter = source.get('quarter', '')
            fiscal_year = source.get('fiscal_year', '')
            period = f"({quarter} {fiscal_year})" if quarter or fiscal_year else ""
            print(f"- {source_name} {period}")

    print("\nRAG system demo complete!")


if __name__ == "__main__":
    main()