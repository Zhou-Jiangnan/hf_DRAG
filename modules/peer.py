from typing import List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from modules.data_types import Datapoint
from modules.knowledge_base import KnowledgeBase
from modules.llm import LLM


class Peer:
    def __init__(self, peer_id: int, llm_url: str, llm_name: str, llm_seed: int):
        """
        Initializes a Peer object.

        Args:
            peer_id: A unique identifier for the peer.
            llm_url: The URL of the Large Language Model (LLM) service.
            llm_name: The name/model identifier of the LLM.
            llm_seed: The seed value for the LLM to ensure reproducibility.
        """
        self.peer_id = peer_id
        self.llm = LLM(llm_url, llm_name, llm_seed)
        self.knowledge_base = KnowledgeBase()
    
    def add_knowledge(self, data_point: Datapoint):
        """
        Adds a new data point to the peer's knowledge base.

        Args:
            data_point: The Datapoint object to be added.
        """
        self.knowledge_base.add(data_point)

    def parse_topic(self, question: str, pre_defined_topics: List[str]) -> Optional[str]:
        """
        Parses the topic of a given question using an LLM.

        Args:
            question: The question to analyze.
            pre_defined_topics: A list of predefined topics to choose from.

        Returns:
            The identified topic from the predefined list, or None if no topic is identified.
        """
        # Construct the prompt for the LLM using a Jinja2 template.
        template_environment = Environment(loader=FileSystemLoader(searchpath="./templates"))
        prompt_template = template_environment.get_template("parse_topic.tmpl")
        llm_prompt = prompt_template.render(question=question, topics=pre_defined_topics)

        # Invoke the LLM to generate a response.
        llm_response = self.llm.generate(llm_prompt)

        # Extract the identified topic from the LLM's response.
        extracted_topic = llm_response.get("topic", None)
        return extracted_topic

    def query(self, question: str, query_confidence_threshold: float) -> Tuple[Optional[str], str, float, bool]:
        """
        Queries the peer's knowledge base for an answer to the given question.

        Args:
            question: The question to answer.
            query_confidence_threshold: The minimum confidence score for a relevant knowledge to be considered a hit.

        Returns:
            A tuple containing:
            - The LLM's generated answer (or None if no answer is found).
            - The most relevant knowledge found in the knowledge base (or "" if nothing relevant is found).
            - The relevance score of the retrieved knowledge.
            - A boolean indicating whether the query was a hit (True) or a miss (False) based on the confidence 
                threshold.
        """
        # Look up relevant knowledge in the local knowledge base using semantic search.
        relevant_knowledge, relevance_score = self.knowledge_base.semantic_search(question)

        # Check if the relevance score meets the confidence threshold.
        if relevance_score > query_confidence_threshold:
            # Construct a prompt for the LLM to generate an answer based on the relevant knowledge.
            template_environment = Environment(loader=FileSystemLoader(searchpath="./templates"))
            answer_template = template_environment.get_template("generate_answer.tmpl")
            llm_prompt = answer_template.render(question=question, context=relevant_knowledge)

            # Invoke the LLM to generate an answer.
            llm_response = self.llm.generate(llm_prompt)

            # Extract the generated answer from the LLM's response.
            generated_answer = llm_response.get("answer", None)
            return generated_answer, relevant_knowledge, relevance_score, True

        # If the confidence threshold is not met, return None for the answer and indicate a query miss.
        return None, relevant_knowledge, relevance_score, False
