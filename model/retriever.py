from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import faiss


class ContextRetriever:
    """
    Semantic Search:
        Uses sentence transformers to create dense embeddings and cosine similarity for matching.
        This is effective for semantic understanding but can be computationally intensive.

        Reimers, N. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."
        arXiv preprint arXiv:1908.10084 (2019).
        https://www.sbert.net/examples/applications/semantic-search/README.html
        https://www.sbert.net/examples/applications/retrieve_rerank/README.html

    Dense Retrieval with FAISS:
        Implements efficient similarity search using Facebook AI's FAISS library.
        This method is particularly useful for large-scale retrieval due to its optimized index structure.

    BM25 Retrieval:
        Implements the BM25 algorithm, which is a strong lexical matching baseline that
        often outperforms more complex neural methods on certain tasks.

        Robertson, Stephen E., Steve Walker, Susan Jones, Micheline M. Hancock-Beaulieu, and Mike Gatford.
        "Okapi at TREC-3." Nist Special Publication Sp 109 (1995): 109.

    Hybrid Retrieval:
        Combines semantic and lexical search with configurable weights, leveraging the strengths of both approaches.

    Ensemble Retrieval:
        Uses reciprocal rank fusion to combine results from multiple retrieval methods,
        often providing more robust results than single methods.

    Usage:
        # Initialize
        retriever = ContextRetriever(local_dataset)

        # Choose a method
        results = retriever.hybrid_retrieval("your question here", top_k=3)
    """
    def __init__(self, local_dataset: List[str], embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'):
        self.local_dataset = local_dataset
        self.embedding_model = SentenceTransformer(embedding_model)

        # Pre-compute embeddings for the dataset
        self.dataset_embeddings = self.embedding_model.encode(local_dataset)

        # Initialize FAISS index
        self.dimension = self.dataset_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.dataset_embeddings.astype('float32'))

        # Initialize BM25
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in local_dataset]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def semantic_search(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Semantic search using sentence transformers and cosine similarity
        """
        # Encode the question
        question_embedding = self.embedding_model.encode([question])[0]

        # Calculate cosine similarity
        similarities = cosine_similarity(
            [question_embedding],
            self.dataset_embeddings
        )[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.local_dataset[i], similarities[i]) for i in top_indices]

    def dense_retrieval_faiss(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Dense retrieval using FAISS for efficient similarity search
        """
        # Encode the question
        question_embedding = self.embedding_model.encode([question]).astype('float32')

        # Search the FAISS index
        distances, indices = self.index.search(question_embedding, top_k)

        return [(self.local_dataset[idx], 1 / (1 + dist)) for dist, idx in zip(distances[0], indices[0])]

    def bm25_retrieval(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Lexical search using BM25 algorithm
        """
        # Tokenize the question
        tokenized_question = word_tokenize(question.lower())

        # Remove stop words
        tokenized_question = [w for w in tokenized_question if w not in self.stop_words]

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_question)

        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.local_dataset[i], scores[i]) for i in top_indices]

    def hybrid_retrieval(self, question: str, top_k: int = 3,
                         semantic_weight: float = 0.7) -> List[Tuple[str, float]]:
        """
        Hybrid retrieval combining semantic and lexical search
        """
        # Get results from both methods
        semantic_results = self.semantic_search(question, top_k=top_k)
        bm25_results = self.bm25_retrieval(question, top_k=top_k)

        # Combine scores
        combined_scores = {}
        for doc, score in semantic_results:
            combined_scores[doc] = score * semantic_weight

        for doc, score in bm25_results:
            if doc in combined_scores:
                combined_scores[doc] += score * (1 - semantic_weight)
            else:
                combined_scores[doc] = score * (1 - semantic_weight)

        # Sort and return top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def ensemble_retrieval(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Ensemble method using reciprocal rank fusion
        """
        # Get results from all methods
        semantic_results = self.semantic_search(question, top_k=top_k)
        dense_results = self.dense_retrieval_faiss(question, top_k=top_k)
        bm25_results = self.bm25_retrieval(question, top_k=top_k)

        # Calculate reciprocal rank fusion scores
        fusion_scores = {}
        k = 60  # fusion parameter

        for rank, (doc, _) in enumerate(semantic_results):
            fusion_scores[doc] = fusion_scores.get(doc, 0) + 1 / (rank + k)

        for rank, (doc, _) in enumerate(dense_results):
            fusion_scores[doc] = fusion_scores.get(doc, 0) + 1 / (rank + k)

        for rank, (doc, _) in enumerate(bm25_results):
            fusion_scores[doc] = fusion_scores.get(doc, 0) + 1 / (rank + k)

        # Sort and return top results
        sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
