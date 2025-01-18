from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearcher:
    """
    Semantic Search:
        Uses sentence transformers to create dense embeddings and cosine similarity for matching.
        This is effective for semantic understanding but can be computationally intensive.

        > Reimers, N. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."
        arXiv preprint arXiv:1908.10084 (2019).

    Example::

        semantic_searcher = SemanticSearcher(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        results = semantic_searcher.search(knowledge_sentences, "your question", top_k=1)

    """
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.device = self.embedding_model.device

    def search(self, knowledge_sentences: List[str], question: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Performs semantic search using sentence transformers and cosine similarity.

        Args:
            knowledge_sentences: A list of sentences representing the knowledge base.
            question: The question sentence.
            top_k: The number of top results to return.

        Returns:
            A list of tuples, where each tuple contains a sentence from the knowledge base and its corresponding 
            similarity score.
        """
        # Compute embeddings for the knowledge base
        knowledge_embeddings = self.embedding_model.encode(
            knowledge_sentences, 
            convert_to_tensor=True, 
            device=self.device
        )
        # Encode the question
        question_embedding = self.embedding_model.encode(
            question, 
            convert_to_tensor=True, 
            device=self.device
        )

        # Calculate cosine similarity using util.cos_sim from sentence_transformers
        similarities = util.cos_sim(knowledge_embeddings, question_embedding)
        similarities = similarities.squeeze()  # Remove extra dimension

        # Get top-k results using torch operations
        top_k = min(top_k, len(knowledge_sentences))  # Ensure top_k doesn't exceed list length
        top_scores, top_indices = similarities.topk(top_k)
        
        # Create a list of (sentence, score) tuples
        top_results = [(knowledge_sentences[idx], float(score)) 
                      for idx, score in zip(top_indices.tolist(), top_scores.tolist())]

        return top_results
