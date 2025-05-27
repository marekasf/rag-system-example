# Salesforce Earnings RAG System

## Problem Description

Build a Retrieval-Augmented Generation (RAG) system for analyzing Salesforce quarterly earnings call transcripts. The system provides Q&A and summarization capabilities through a conversational interface, enabling analysts to quickly extract insights from financial documents.

**Key Requirements:**
- Answer questions about earnings data and company performance
- Generate summaries of strategies, risks, and financial results
- Handle both structured and unstructured document content
- Support queries like "most recent earnings call" and "risks over time"

## Implementation Overview

**Architecture:** Conversational RAG with re-ranking
- Document ingestion with PDF processing and metadata extraction
- Token-based chunking with LLM-enhanced metadata
- Vector embeddings using HuggingFace (with OpenAI fallback)
- FAISS vector store for similarity search
- Simple re-ranking based on keyword matching
- OpenAI GPT-4 for response generation

**Technology Stack:**
- **Vector Database:** FAISS (local storage)
- **LLM:** OpenAI GPT-4 for response generation and metadata extraction
- **Embeddings:** HuggingFace sentence-transformers (primary), OpenAI (alternative)
- **Framework:** LangChain for RAG pipeline orchestration
- **UI:** Streamlit for conversational interface

**System Components:**
1. **DocumentIngestion** - PDF/text loading with configurable modes
2. **DocumentProcessor** - Chunking and LLM-based metadata extraction
3. **EmbeddingGenerator** - Vector embedding creation with caching
4. **VectorStore** - FAISS-based similarity search with metadata filtering
5. **RetrievalController** - Semantic search with simple re-ranking
6. **RAGController** - End-to-end query processing and response generation

## Key Design Decisions

**PDF Processing:** Single-document mode to preserve context across pages rather than page-by-page splitting

**Chunking Strategy:** Token-based splitting (5000 chars, 400 overlap) optimized for LLM context windows

**Embedding Approach:** HuggingFace models for cost-effective local processing with OpenAI as premium alternative

**Metadata Enhancement:** LLM-based extraction of quarters, fiscal years, speakers, and topics with disk caching

**Vector Storage:** FAISS for local deployment and rapid prototyping vs. cloud solutions

**Re-ranking:** Simple keyword-based boosting for improved relevance without complex ML models

**Conversational Memory:** Basic history tracking suitable for POC requirements

## Evaluation

Built-in evaluation system tests sample questions from technical requirements:
- Retrieval quality metrics (relevance ratio, response time)
- Response quality assessment (term coverage, grounding ratio)
- System performance statistics

## Usage

```bash
# Run document ingestion
python src/amk/controllers/ingestion_controller.py

# Start Streamlit UI
streamlit run src/ui/rag_ui.py

# Run evaluation
python src/amk/evaluation/RAG_evaluator.py
```

**Environment:** Requires `OPENAI_API_KEY` for LLM functionality. Documents placed in `resources/amk/data/` directory.