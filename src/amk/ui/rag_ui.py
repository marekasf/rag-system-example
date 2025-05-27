"""
Streamlit UI for Salesforce Earnings RAG System
A simple conversational interface for querying Salesforce earnings call transcripts.
"""

import sys
from pathlib import Path
base_path = Path(__file__).parent.parent.parent.parent
# Add the src directory to the Python path
sys.path.append(str(base_path / "src"))


import streamlit as st
import os
from typing import Dict, Any
import logging



# Import RAG system components
from amk.controllers.RAG_controller import RAGController
from amk.controllers.retrieval_controller import RetrievalController
from amk.processing.embedding_generator import EmbeddingGenerator
from amk.storage.faiss_storage import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Salesforce Earnings RAG System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_rag_system():
    """
    Initialize the RAG system components.
    Uses Streamlit caching to avoid reinitializing on every interaction.

    Returns:
        RAGController instance or None if initialization fails
    """
    try:
        # Set up paths

        resources_path = base_path / "resources" / "amk"
        index_path = resources_path / "faiss"

        # Check if vector store exists
        if not (index_path / "vector_store.index").exists():
            st.error(f"Vector store not found at {index_path}. Please run the ingestion process first.")
            return None

        # Load vector store
        vector_store = VectorStore.load_index(index_path)

        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()

        # Initialize retrieval controller
        retrieval_controller = RetrievalController(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            top_k=5
        )

        # Initialize RAG controller
        rag_controller = RAGController(
            retrieval_controller=retrieval_controller,
            top_k=5
        )

        return rag_controller

    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        logger.error(f"RAG system initialization failed: {str(e)}")
        return None

def display_system_stats(rag_controller: RAGController):
    """
    Display system statistics in the sidebar.

    Args:
        rag_controller: Initialized RAGController instance
    """
    with st.sidebar:
        st.header("📊 System Statistics")

        try:
            stats = rag_controller.get_system_stats()

            st.metric("Total Documents", stats.get('total_documents', 'Unknown'))
            st.metric("Unique Sources", stats.get('unique_sources', 'Unknown'))
            st.metric("Conversation Turns", stats.get('conversation_turns', 0))

            # Show available quarters if any
            quarters = stats.get('unique_quarters', [])
            if quarters:
                st.write("**Available Quarters:**")
                for quarter in quarters:
                    st.write(f"• {quarter}")

            # Show fiscal years if any
            fiscal_years = stats.get('fiscal_years', [])
            if fiscal_years:
                st.write("**Fiscal Years:**")
                for year in fiscal_years:
                    st.write(f"• {year}")

        except Exception as e:
            st.error(f"Error loading system stats: {str(e)}")

def display_sample_questions():
    """
    Display sample questions users can ask.
    """
    with st.sidebar:
        st.header("💡 Sample Questions")

        sample_questions = [
            "When was the most recent earnings call?",
            "What are the risks that Salesforce has faced?",
            "Can you summarize Salesforce's strategy in 2023?",
            "How many earnings call documents do you have indexed?",
            "What were the financial results in Q3 2023?",
            "Who are the key speakers in earnings calls?",
            "What are the main topics discussed in recent calls?"
        ]

        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_q_{i}"):
                st.session_state.current_question = question

def format_response_with_sources(result: Dict[str, Any]) -> str:
    """
    Format the response with sources for display.

    Args:
        result: Result dictionary from RAG controller

    Returns:
        Formatted response string
    """
    response = result.get('response', 'No response generated.')

    # TODO: Implement better source formatting and citation
    # For POC, just add basic source information
    sources = result.get('sources', [])
    if sources:
        response += "\n\n**Sources:**\n"
        for i, source in enumerate(sources[:3]):  # Show top 3 sources
            source_name = source.get('source', 'Unknown')
            quarter = source.get('quarter', '')
            fiscal_year = source.get('fiscal_year', '')
            period = f"({quarter} {fiscal_year})" if quarter or fiscal_year else ""
            response += f"{i+1}. {source_name} {period}\n"

    return response

def display_conversation_history():
    """
    Display conversation history in the sidebar.
    """
    with st.sidebar:
        st.header("💬 Conversation History")

        if 'conversation_history' in st.session_state and st.session_state.conversation_history:
            # Show last few conversations
            # TODO: Implement pagination for longer histories
            recent_history = st.session_state.conversation_history[-5:]  # Last 5 conversations

            for i, turn in enumerate(reversed(recent_history)):
                with st.expander(f"Q: {turn['question'][:50]}{'...' if len(turn['question']) > 50 else ''}"):
                    st.write("**Question:**")
                    st.write(turn['question'])
                    st.write("**Answer:**")
                    st.write(turn['answer'])
        else:
            st.write("No conversation history yet. Start asking questions!")

        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            if 'rag_controller' in st.session_state and st.session_state.rag_controller:
                st.session_state.rag_controller.clear_conversation_history()
            st.rerun()

def main():
    """
    Main Streamlit application.
    """
    st.title("📊 Salesforce Earnings RAG System")
    st.markdown("Ask questions about Salesforce's quarterly earnings call transcripts")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'rag_controller' not in st.session_state:
        st.session_state.rag_controller = None

    # Initialize RAG system
    if st.session_state.rag_controller is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_controller = initialize_rag_system()

    # Check if system initialized successfully
    if st.session_state.rag_controller is None:
        st.error("❌ RAG system failed to initialize. Please check your setup.")
        st.info("Make sure you have:")
        st.info("1. Run the document ingestion process")
        st.info("2. Set your OPENAI_API_KEY environment variable")
        st.info("3. Have documents in the resources/amk/data directory")
        return

    # Display system stats and sample questions in sidebar
    display_system_stats(st.session_state.rag_controller)
    display_sample_questions()
    display_conversation_history()

    # Main chat interface
    st.header("💬 Ask a Question")

    # Check if a sample question was selected
    if 'current_question' in st.session_state:
        user_question = st.session_state.current_question
        del st.session_state.current_question  # Clear the selected question
    else:
        user_question = ""

    # Question input
    question = st.text_area(
        "Enter your question about Salesforce earnings:",
        value=user_question,
        height=100,
        placeholder="e.g., What were Salesforce's financial results in Q3 2023?"
    )

    # TODO: Add advanced options like filtering by quarter/year
    # For POC, we'll just implement basic Q&A

    col1, col2 = st.columns([1, 4])

    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary")

    with col2:
        # TODO: Add summarization functionality
        summary_button = st.button("📝 Generate Summary", disabled=True)
        if summary_button:
            st.info("Summary feature coming soon!")

    # Process question
    if ask_button and question.strip():
        with st.spinner("Searching documents and generating response..."):
            try:
                # Generate response using RAG controller
                result = st.session_state.rag_controller.generate_response(
                    query=question,
                    include_sources=True
                )

                if result.get('success', False):
                    # Format and display response
                    formatted_response = format_response_with_sources(result)

                    st.success("✅ Response generated successfully!")

                    # Display the response
                    st.markdown("### 🤖 Answer:")
                    st.markdown(formatted_response)

                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'question': question,
                        'answer': result.get('response', ''),
                        'sources': result.get('sources', [])
                    })

                else:
                    st.error("❌ Failed to generate response. Please try again.")

            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                logger.error(f"Error processing question: {str(e)}")

    elif ask_button and not question.strip():
        st.warning("⚠️ Please enter a question before clicking Ask.")

    # Display recent conversation in main area
    if st.session_state.conversation_history:
        st.header("📝 Recent Conversations")

        # Show last 3 conversations in main area
        # TODO: Implement better conversation display with expandable sections
        recent_convos = st.session_state.conversation_history[-3:]

        for i, convo in enumerate(reversed(recent_convos)):
            with st.expander(f"💬 {convo['question'][:80]}{'...' if len(convo['question']) > 80 else ''}", expanded=(i == 0)):
                st.markdown("**Question:**")
                st.write(convo['question'])
                st.markdown("**Answer:**")
                st.write(convo['answer'])

                # TODO: Add functionality to regenerate response, rate response, etc.

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("❌ OpenAI API key not found!")
        st.info("Please set the OPENAI_API_KEY environment variable and restart the app.")
        st.stop()

    main()