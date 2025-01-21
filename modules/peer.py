from typing import List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from modules.data_types import Datapoint
from modules.knowledge_base import KnowledgeBase
from modules.llm import LLM


class Peer:
    def __init__(self, peer_id: int, llm_url: str, llm_name: str, llm_seed: int):
        self.peer_id = peer_id
        self.llm = LLM(llm_url, llm_name, llm_seed)
        self.knowledge_base = KnowledgeBase()
    
    def add_knowledge(self, data_point: Datapoint):
        self.knowledge_base.add(data_point)

    def parse_topic(self, question: str, pre_defined_topics: List[str]):
        """Parse the topic of the question"""
        # Construct prompt
        env = Environment(loader=FileSystemLoader(searchpath="./templates"))
        prompt_tmpl = env.get_template("parse_topic.tmpl")
        prompt = prompt_tmpl.render(question=question, topics=pre_defined_topics)
        # Invoke LLM
        response_json = self.llm.generate(prompt)
        topic = response_json.get("topic", None)
        return topic

    def query(self, question: str, query_confidence_threshold: float) -> Tuple[Optional[str], str, float]:
        # look up in local knowledge base
        relevant_knowledge, relevant_score = self.knowledge_base.semantic_search(question)
        # if find out a valid answer, return an answer
        if relevant_score > query_confidence_threshold:
            # Construct prompt
            env = Environment(loader=FileSystemLoader(searchpath="./templates"))
            prompt_tmpl = env.get_template("generate_answer.tmpl")
            prompt = prompt_tmpl.render(question=question, context=relevant_knowledge)
            # Invoke LLM
            response_json = self.llm.generate(prompt)
            answer = response_json.get("answer", None)
            return answer, relevant_knowledge, relevant_score
        # if cannot find a valid answer
        return None, relevant_knowledge, relevant_score
