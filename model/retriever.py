from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ContextRetriever:
    """
    Semantic Search:
        Uses sentence transformers to create dense embeddings and cosine similarity for matching.
        This is effective for semantic understanding but can be computationally intensive.

        > Reimers, N. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."
        arXiv preprint arXiv:1908.10084 (2019).

    Usage:
    ```python
    retriever = ContextRetriever(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    results = retriever.semantic_search(local_sentences, "your question", top_k=1)
    ```
    """
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)

    def semantic_search(self, local_sentences: List[str], question: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Semantic search using sentence transformers and cosine similarity
        """
        # Compute embeddings for the dataset
        dataset_embeddings = self.embedding_model.encode(local_sentences)
        # Encode the question
        question_embedding = self.embedding_model.encode([question])[0]

        # Calculate cosine similarity
        similarities = cosine_similarity([question_embedding], dataset_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_results = [(local_sentences[i], similarities[i]) for i in top_indices]

        return top_results
