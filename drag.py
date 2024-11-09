import json
from dataclasses import dataclass
import logging
import random
import requests
from typing import List, Dict, Optional

from datasets import load_dataset
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from jsonargparse import ArgumentParser

from model.evaluator import Evaluator
from model.retriever import ContextRetriever

logger = logging.getLogger(__file__)


@dataclass
class RAGAnswer:
    text: str
    confidence: float


class Datapoint:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer
    
    def __repr__(self):
        return f"[Question]:{self.question}[Answer]:{self.answer}"


class RAGNode:
    def __init__(
            self,
            node_id: int,
            llm_api_endpoint: str,
            llm_name: str,
            confidence_threshold: float = 0.7,
            num_retrieval_answers: int = 3
    ):
        self.node_id = node_id
        self.llm_api_endpoint = llm_api_endpoint
        self.llm_name = llm_name
        self.confidence_threshold = confidence_threshold
        self.num_retrieval_answers = num_retrieval_answers
        self.retriever = ContextRetriever()
        self.local_dataset: List[Datapoint] = []

    def add_to_local_dataset(self, datapoint: Datapoint):
        """Add a new datapoint to local RAG dataset"""
        self.local_dataset.append(datapoint)
    
    def execute_llm(self, prompt: str) -> RAGAnswer:
        body_data = {
            "model": self.llm_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        response = requests.post(
            self.llm_api_endpoint,
            json=body_data,
        )
        response_text = response.json()["response"]

        # print(f"Debug response_text: {response_text}")
        response_json = json.loads(response_text)

        # Parse answer and confidence
        answer = response_json.get("answer", "None")
        confidence = response_json.get("confidence", 0.0)
        confidence = float(confidence)
        # print(f"Debug [question]: {question} [answer]: {answer}; [confidence]: {confidence}")
        return RAGAnswer(answer, confidence)

    def generate_answer(self, question: str, retrieved_answers: Dict[int, RAGAnswer]|None = None) -> RAGAnswer:
        """Generate answer using local dataset and LLM"""
        # Find most relevant context from local dataset
        relevant_context = self.retrieve_local_context(question)

        # Construct prompt with retrieved answers
        if retrieved_answers is not None:
            retrieved_context = "\n".join(
                f"Answer from {node_id} (confidence: {ans.confidence}):"
                f" {ans.text}"
                for node_id, ans in retrieved_answers
            )
            relevant_context = relevant_context + "\n" + retrieved_context

        # Construct prompt with context
        prompt = f"""
        Question:
        {question}

        Context:
        {relevant_context}

        Please provide an answer based on the context above and rate your confidence from 0.0 to 1.0.
        Respond using JSON with keys: `answer`, `confidence`."""

        # Execute LLM
        rag_answer = self.execute_llm(prompt)
        return rag_answer

    def retrieve_local_context(self, question: str) -> str:
        """Retrieve relevant context from local dataset"""
        relevant_context = self.retriever.semantic_search(self.local_dataset, question)[0]
        return str(relevant_context)


class DistributedRAGSystem:
    def __init__(
            self, 
            llm_api_endpoint: str, 
            llm_name: str, 
            num_nodes: int, 
            datapoints: List[Datapoint], 
            confidence_threshold: float
        ):

        self.confidence_threshold = confidence_threshold
        # Initialize nodes
        self.nodes: Dict[int, RAGNode] = {}
        for node_id in range(num_nodes):
            self.nodes[node_id] = RAGNode(
                node_id=node_id, 
                llm_api_endpoint=llm_api_endpoint, 
                llm_name=llm_name,
            )
        
        # Initialize data points
        for i, datapoint in enumerate(datapoints):
            node_idx = i % num_nodes
            self.nodes[node_idx].add_to_local_dataset(datapoint)

    
    def query(self, question: str, selected_node_id: Optional[str] = None) -> RAGAnswer:
        if selected_node_id is None:
            selected_node_id = random.choice(list(self.nodes.keys()))
        
        selected_node = self.nodes[selected_node_id]

        # Get answer from starting node
        original_answer = selected_node.generate_answer(question)

        # If original answer is confident enough, return
        if original_answer.confidence >= self.confidence_threshold:
            return original_answer

        # Select random subset of neighbors to retrieval
        selected_neighbor_ids = random.sample(list(self.nodes.keys()), 3)
        
        retrieved_answers: Dict[int, RAGAnswer] = {}
        
        # Get answers from selected neighbors
        for neighbor_id in selected_neighbor_ids:
            neighbor = self.nodes[neighbor_id]
            retrieved_answers[neighbor_id] = neighbor.generate_answer(question)

        # TODO: Rank answers by confidence and select top k
        final_answer = selected_node.generate_answer(question, retrieved_answers)
        
        return final_answer


def get_dict_val(dict_item, dot_keys):
    """Get nested value from the dict"""
    keys = dot_keys.split(".")
    val = dict_item
    for key in keys:
        val = val[key]
    return val


def run_simulation(llm_api_endpoint: str, llm_name: str, data_config: dict, num_nodes: int = 10):
    """Run evaluation on a HuggingFace dataset"""
    # Load Huggingface dataset
    dataset = load_dataset(**data_config["load"])
    datapoints: List[Datapoint] = []

    # random pick 20 samples for testing
    for i, item in enumerate(dataset.take(20)):
        question = get_dict_val(item, data_config["question_path"])
        answer = get_dict_val(item, data_config["answer_path"])
        datapoint = Datapoint(question, answer)
        datapoints.append(datapoint)

    # Initialize distributed RAG system
    drag_system = DistributedRAGSystem(
        num_nodes=num_nodes, 
        datapoints=datapoints, 
        llm_api_endpoint=llm_api_endpoint, 
        llm_name=llm_name
    )

    # Run evaluation
    evaluation_dataset = EvaluationDataset()

    for datapoint in datapoints:
        # Each question is processed by a random node
        drag_answer = drag_system.query(datapoint)

        # DeepEval test case
        test_case = LLMTestCase(
            input=datapoint.question,
            actual_output=drag_answer.text,
            expected_output=datapoint.answer,
        )
        evaluation_dataset.add_test_case(test_case)

    # Calculate metrics
    my_evaluator = Evaluator()
    my_evaluator.evaluate(evaluation_dataset)


def main():
    # parse arguments
    parser = ArgumentParser(default_config_files=["./config/default.yaml"])
    parser.add_argument("--model.endpoint", type=str)
    parser.add_argument("--model.name", type=str)

    parser.add_argument("--data.load.path", type=str)
    parser.add_argument("--data.load.name", type=str)
    parser.add_argument("--data.load.split", type=str)
    parser.add_argument("--data.question_path", type=str)
    parser.add_argument("--data.answer_path", type=str)

    cfg = parser.parse_args()

    # run evaluation
    run_simulation(cfg.model.endpoint, cfg.model.name, cfg.data.as_dict())


if __name__ == "__main__":
    main()
