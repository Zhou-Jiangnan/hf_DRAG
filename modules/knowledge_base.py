from typing import List

from modules.data_types import Datapoint
from modules.semantic_searcher import SemanticSearcher


class KnowledgeBase:
    def __init__(self):
        self.knowledges: List[Datapoint] = []
        self.semantic_searcher = SemanticSearcher()

    def add(self, data_point: Datapoint):
        self.knowledges.append(data_point)

    def semantic_search(self, question: str):
        """Search relevant knowledge from local knowledge base"""
        if len(self.knowledges) == 0:
            return "", 0.0

        knowledge_sentences = [f"[Question]: {data_point.question} [Answer]: {data_point.answer}"
                           for data_point in self.knowledges]
        relevant_knowledge, relevant_score = self.semantic_searcher.search(knowledge_sentences, question)[0]
        return relevant_knowledge, relevant_score
