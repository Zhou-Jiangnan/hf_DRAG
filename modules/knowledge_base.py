from typing import List

from modules.data_types import Datapoint
from modules.semantic_searcher import SemanticSearcher


class KnowledgeBase:
    def __init__(self):
        """
        Initializes an empty KnowledgeBase.
        """
        self.data_points: List[Datapoint] = []
        self.semantic_searcher = SemanticSearcher()

    def add(self, data_point: Datapoint):
        """
        Adds a data point to the knowledge base.

        Args:
            data_point: The Datapoint object to add.
        """
        self.data_points.append(data_point)

    def semantic_search(self, query: str) -> tuple[str, float]:
        """
        Performs a semantic search to find relevant knowledge within the knowledge base.

        Args:
            query: The question or search query.

        Returns:
            A tuple containing:
            - The most relevant knowledge found (as a formatted string).
            - The relevance score (a float between 0.0 and 1.0).
        """
        if not self.data_points:
            return "", 0.0

        # Format the knowledge base entries for semantic search.
        formatted_knowledge_entries = [
            f"# Question:\n{data_point.question}\n# Correct Answer:\n{data_point.answer}\n"
            for data_point in self.data_points
        ]

        # Perform the semantic search.
        # Assuming the search method of SemanticSearcher returns a list of tuples, each containing (text, score)
        search_results = self.semantic_searcher.search(formatted_knowledge_entries, query)

        # Get the top result (most relevant).
        most_relevant_knowledge, relevance_score = search_results[0]

        return most_relevant_knowledge, relevance_score
