"""
RAGEvaluator module for evaluating the performance of a RAG system.
This module provides functionality to assess the accuracy, relevance, and completeness
of Q&A and summarization responses from the Salesforce earnings call RAG system.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Core document type
from langchain_core.documents import Document

# Import our RAG system components
from amk.storage.faiss_storage import VectorStore
from amk.processing.embedding_generator import EmbeddingGenerator
from amk.controllers.retrieval_controller import RetrievalController

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    A class for evaluating the performance of a RAG system.

    This class handles:
    - Testing the system with predefined sample questions
    - Measuring retrieval quality (relevance of retrieved documents)
    - Evaluating response quality (accuracy and completeness)
    - Generating evaluation reports with metrics
    - Simple LLM-as-judge evaluation for POC purposes
    """

    def __init__(
            self,
            retrieval_controller: RetrievalController,
            embedding_generator: EmbeddingGenerator,
            vector_store: VectorStore
    ):
        """
        Initialize the RAGEvaluator with system components.

        Args:
            retrieval_controller: RetrievalController instance for document retrieval
            embedding_generator: EmbeddingGenerator for embedding queries
            vector_store: VectorStore for accessing document statistics
        """
        self.retrieval_controller = retrieval_controller
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

        # Sample questions from the technical assessment requirements
        self.sample_questions = [
            "When was the most recent earnings call?",
            "What are the risks that Salesforce has faced, and how have they changed over time?",
            "Can you summarize Salesforce's strategy at the beginning of 2023?",
            "How many earnings call documents do you have indexed?",
            "How many pages are in the most recent earnings call?"
        ]

        logger.info("Initialized RAGEvaluator with predefined sample questions")

    def evaluate_retrieval_quality(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluate the quality of document retrieval for a given query.

        This method assesses how well the retrieval system finds relevant documents
        without considering the final response generation.

        Args:
            query: User query string
            top_k: Number of documents to retrieve for evaluation

        Returns:
            Dictionary containing retrieval quality metrics
        """
        logger.info(f"Evaluating retrieval quality for query: '{query}'")

        start_time = time.time()

        # Retrieve documents using the retrieval controller
        retrieved_docs = self.retrieval_controller.retrieve(query, top_k=top_k)

        retrieval_time = time.time() - start_time

        # Simple metrics for POC - in production, would use more sophisticated evaluation
        metrics = {
            "query": query,
            "num_retrieved": len(retrieved_docs),
            "retrieval_time_seconds": retrieval_time,
            "documents_found": len(retrieved_docs) > 0,
        }

        # Basic relevance assessment - check if retrieved docs contain query terms
        # TODO: Implement more sophisticated relevance scoring using:
        # - BERT-based semantic similarity
        # - Human-annotated relevance scores
        # - Domain-specific relevance criteria for financial documents

        query_terms = query.lower().split()
        relevant_docs = 0

        for doc in retrieved_docs:
            content_lower = doc.page_content.lower()
            # Simple keyword matching for POC
            if any(term in content_lower for term in query_terms):
                relevant_docs += 1

        metrics["relevance_ratio"] = relevant_docs / len(retrieved_docs) if retrieved_docs else 0
        metrics["avg_doc_length"] = sum(len(doc.page_content) for doc in retrieved_docs) / len(
            retrieved_docs) if retrieved_docs else 0

        logger.info(f"Retrieval evaluation completed: {relevant_docs}/{len(retrieved_docs)} docs deemed relevant")

        return metrics

    def evaluate_response_quality(self, query: str, response: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        """
        Evaluate the quality of the generated response.

        This is a simple implementation for POC. In production, would use more
        sophisticated evaluation methods like RAGAS, LLM-as-judge, or human evaluation.

        Args:
            query: Original user query
            response: Generated response from the RAG system
            retrieved_docs: Documents that were retrieved for this query

        Returns:
            Dictionary containing response quality metrics
        """
        logger.info(f"Evaluating response quality for query: '{query}'")

        # Simple metrics for POC
        metrics = {
            "query": query,
            "response_length": len(response),
            "response_not_empty": len(response.strip()) > 0,
        }

        # Check if response addresses the query (basic keyword matching)
        query_terms = set(word.lower() for word in query.split() if len(word) > 3)
        response_terms = set(word.lower() for word in response.split() if len(word) > 3)

        # Calculate term overlap as a simple relevance metric
        if query_terms:
            term_overlap = len(query_terms.intersection(response_terms)) / len(query_terms)
            metrics["query_term_coverage"] = term_overlap
        else:
            metrics["query_term_coverage"] = 0

        # Check if response is grounded in retrieved documents
        # TODO: Implement more sophisticated grounding evaluation:
        # - Fact verification against source documents
        # - Citation accuracy checking
        # - Hallucination detection

        grounded_sentences = 0
        response_sentences = response.split('.')

        for sentence in response_sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue

            # Check if sentence content appears in any retrieved document
            sentence_lower = sentence.lower()
            for doc in retrieved_docs:
                doc_content_lower = doc.page_content.lower()
                # Very simple grounding check - look for partial matches
                if any(word in doc_content_lower for word in sentence_lower.split() if len(word) > 4):
                    grounded_sentences += 1
                    break

        total_sentences = len([s for s in response_sentences if len(s.strip()) >= 10])
        metrics["grounding_ratio"] = grounded_sentences / total_sentences if total_sentences > 0 else 0

        logger.info(f"Response evaluation completed: {grounded_sentences}/{total_sentences} sentences appear grounded")

        return metrics

    def run_sample_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation on all predefined sample questions.

        This method tests the RAG system with the sample questions from the
        technical assessment and generates a comprehensive evaluation report.

        Returns:
            Dictionary containing complete evaluation results
        """
        logger.info("Starting evaluation with sample questions from technical assessment")

        evaluation_results = {
            "evaluation_timestamp": time.time(),
            "total_questions": len(self.sample_questions),
            "individual_results": [],
            "aggregate_metrics": {}
        }

        # Evaluate each sample question
        for i, question in enumerate(self.sample_questions):
            logger.info(f"Evaluating question {i + 1}/{len(self.sample_questions)}: {question}")

            start_time = time.time()

            # Step 1: Evaluate retrieval quality
            retrieval_metrics = self.evaluate_retrieval_quality(question, top_k=3)

            # Step 2: Retrieve documents for response generation
            retrieved_docs = self.retrieval_controller.retrieve(question, top_k=3)

            # Step 3: Generate a simple response (for POC, just concatenate top docs)
            # TODO: Integrate with actual response generation component
            # In production, this would call the LLM with proper prompt engineering
            if retrieved_docs:
                # Simple response generation for POC - just use content from top document
                response = f"Based on the earnings call documents: {retrieved_docs[0].page_content[:500]}..."
            else:
                response = "I couldn't find relevant information in the earnings call documents to answer this question."

            # Step 4: Evaluate response quality
            response_metrics = self.evaluate_response_quality(question, response, retrieved_docs)

            total_time = time.time() - start_time

            # Combine results for this question
            question_result = {
                "question": question,
                "total_evaluation_time": total_time,
                "retrieval_metrics": retrieval_metrics,
                "response_metrics": response_metrics,
                "sample_response": response[:200] + "..." if len(response) > 200 else response
            }

            evaluation_results["individual_results"].append(question_result)

            logger.info(f"Completed evaluation for question {i + 1}")

        # Calculate aggregate metrics
        evaluation_results["aggregate_metrics"] = self._calculate_aggregate_metrics(
            evaluation_results["individual_results"]
        )

        logger.info("Sample evaluation completed")
        return evaluation_results

    def _calculate_aggregate_metrics(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all evaluated questions.

        Args:
            individual_results: List of individual question evaluation results

        Returns:
            Dictionary containing aggregate metrics
        """
        if not individual_results:
            return {}

        # Extract metrics from individual results
        retrieval_times = [r["retrieval_metrics"]["retrieval_time_seconds"] for r in individual_results]
        relevance_ratios = [r["retrieval_metrics"]["relevance_ratio"] for r in individual_results]
        documents_found_count = sum(1 for r in individual_results if r["retrieval_metrics"]["documents_found"])

        response_lengths = [r["response_metrics"]["response_length"] for r in individual_results]
        term_coverages = [r["response_metrics"]["query_term_coverage"] for r in individual_results]
        grounding_ratios = [r["response_metrics"]["grounding_ratio"] for r in individual_results]

        # Calculate aggregate statistics
        aggregate_metrics = {
            # Retrieval metrics
            "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times),
            "avg_relevance_ratio": sum(relevance_ratios) / len(relevance_ratios),
            "documents_found_percentage": (documents_found_count / len(individual_results)) * 100,

            # Response metrics
            "avg_response_length": sum(response_lengths) / len(response_lengths),
            "avg_term_coverage": sum(term_coverages) / len(term_coverages),
            "avg_grounding_ratio": sum(grounding_ratios) / len(grounding_ratios),

            # Overall system metrics
            "total_questions_evaluated": len(individual_results),
            "questions_with_responses": sum(
                1 for r in individual_results if r["response_metrics"]["response_not_empty"])
        }

        return aggregate_metrics

    def save_evaluation_report(self, evaluation_results: Dict[str, Any], output_path: Optional[Path] = None) -> None:
        """
        Save evaluation results to a JSON file for later analysis.

        Args:
            evaluation_results: Results from run_sample_evaluation()
            output_path: Path to save the report file
        """
        if output_path is None:
            resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
            output_path = resources_path / "evaluation_report.json"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert timestamp to readable format
        evaluation_results["evaluation_timestamp_readable"] = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(evaluation_results["evaluation_timestamp"])
        )

        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation report saved to: {output_path}")

    def print_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> None:
        """
        Print a human-readable summary of evaluation results.

        Args:
            evaluation_results: Results from run_sample_evaluation()
        """
        print("\n" + "=" * 60)
        print("RAG SYSTEM EVALUATION SUMMARY")
        print("=" * 60)

        aggregate = evaluation_results["aggregate_metrics"]

        print(f"\nTotal Questions Evaluated: {aggregate['total_questions_evaluated']}")
        print(f"Questions with Responses: {aggregate['questions_with_responses']}")

        print(f"\nRETRIEVAL PERFORMANCE:")
        print(f"  Average Retrieval Time: {aggregate['avg_retrieval_time']:.3f} seconds")
        print(f"  Documents Found: {aggregate['documents_found_percentage']:.1f}% of queries")
        print(f"  Average Relevance Ratio: {aggregate['avg_relevance_ratio']:.3f}")

        print(f"\nRESPONSE QUALITY:")
        print(f"  Average Response Length: {aggregate['avg_response_length']:.0f} characters")
        print(f"  Average Term Coverage: {aggregate['avg_term_coverage']:.3f}")
        print(f"  Average Grounding Ratio: {aggregate['avg_grounding_ratio']:.3f}")

        print(f"\nSAMPLE QUESTIONS TESTED:")
        for i, result in enumerate(evaluation_results["individual_results"], 1):
            print(f"  {i}. {result['question']}")

        print("\n" + "=" * 60)


def main():
    """
    Example usage of the RAGEvaluator class.
    This function demonstrates how to evaluate a complete RAG system
    using the sample questions from the technical assessment.
    """
    # Import required classes for the example
    from pathlib import Path
    from amk.processing.embedding_generator import EmbeddingGenerator
    from amk.storage.faiss_storage import VectorStore
    from amk.controllers.retrieval_controller import RetrievalController

    print("Starting RAG System Evaluation")
    print("=" * 50)

    # Set up paths
    resources_path = Path(__file__).parent.parent.parent.parent / "resources" / "amk"
    index_path = resources_path / "faiss"

    # 1. Load the vector store
    print("Loading vector store...")
    vector_store = VectorStore.load_index(index_path)
    print(f"Loaded vector store with {len(vector_store.documents)} documents")

    # 2. Initialize embedding generator
    print("Initializing embedding generator...")
    embedding_generator = EmbeddingGenerator()

    # 3. Initialize retrieval controller
    print("Initializing retrieval controller...")
    retrieval_controller = RetrievalController(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        top_k=3  # Retrieve top 3 documents for evaluation
    )

    # 4. Initialize the evaluator
    print("Initializing RAG evaluator...")
    evaluator = RAGEvaluator(
        retrieval_controller=retrieval_controller,
        embedding_generator=embedding_generator,
        vector_store=vector_store
    )

    # 5. Run evaluation on sample questions
    print("\nRunning evaluation on sample questions...")
    evaluation_results = evaluator.run_sample_evaluation()

    # 6. Print summary of results
    evaluator.print_evaluation_summary(evaluation_results)

    # 7. Save detailed evaluation report
    print("\nSaving evaluation report...")
    evaluator.save_evaluation_report(evaluation_results)

    # 8. Optional: Test individual question evaluation
    print("\nTesting individual question evaluation...")
    test_query = "What were Salesforce's key financial metrics in the most recent quarter?"

    retrieval_metrics = evaluator.evaluate_retrieval_quality(test_query)
    print(f"Retrieval metrics for test query: {retrieval_metrics}")

    # TODO: Future enhancements for more comprehensive evaluation:
    # - Implement RAGAS evaluation framework integration
    # - Add human annotation capabilities for ground truth creation
    # - Include LLM-as-judge evaluation for response quality
    # - Add domain-specific metrics for financial document accuracy
    # - Implement comparative evaluation against baseline systems
    # - Add evaluation for different types of queries (factual, analytical, summarization)
    # - Include evaluation of citation accuracy and source attribution
    # - Add performance benchmarking across different model configurations

    print("\nRAG system evaluation complete!")


if __name__ == "__main__":
    main()