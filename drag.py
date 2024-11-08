import json
from dataclasses import dataclass
import logging
import random
import requests
from typing import List

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
    source_id: str  # Identifier for the node that generated this answer


@dataclass
class RetrievalRequest:
    question: str
    source_id: str


class Datapoint:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer
    
    def __repr__(self):
        return f"[Question]:{self.question}[Answer]:{self.answer}"


class RAGNode:
    def __init__(
            self,
            node_id: str,
            llm_api_endpoint: str,
            llm_name: str,
            confidence_threshold: float = 0.7,
            num_retrievals: int = 3
    ):
        self.node_id = node_id
        self.llm_api_endpoint = llm_api_endpoint
        self.llm_name = llm_name
        self.confidence_threshold = confidence_threshold
        self.num_retrievals = num_retrievals
        self.retriever = ContextRetriever()
        self.local_dataset: List[Datapoint] = []

    def add_to_local_dataset(self, datapoint: Datapoint):
        """Add a new datapoint to local RAG dataset"""
        self.local_dataset.append(datapoint)

    def llm_post(self, prompt):
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
        return response_text

    def process_question(self, question: str, peer_nodes: List[str]) -> str:
        """Process a question using DRAG workflow"""
        # Generate initial answer
        local_answer = self.generate_answer(question)

        if local_answer.confidence >= self.confidence_threshold:
            return local_answer.text

        # Send retrieval requests to peers
        retrieval_request = RetrievalRequest(question, self.node_id)
        peer_answers = self.get_peer_answers(retrieval_request, peer_nodes)

        # Combine all answers including local
        all_answers = [local_answer] + peer_answers

        # Rank answers by confidence and select top k
        ranked_answers = sorted(
            all_answers,
            key=lambda x: x.confidence,
            reverse=True
        )
        selected_answers = ranked_answers[:self.num_retrievals]

        # Generate final answer using selected answers as context
        final_answer = self.generate_final_answer(question, selected_answers)
        return final_answer

    def generate_answer(self, question: str) -> RAGAnswer:
        """Generate answer using local dataset and LLM"""
        # Find most relevant context from local dataset
        relevant_context = self.retrieve_local_context(question)

        # Construct prompt with context
        prompt = f"""
        Question: {question}
        Context: {relevant_context}
        Please provide an answer based on the context above and rate your confidence from 0.0 to 1.0.
        Respond using JSON with keys: `answer`, `confidence`."""

        # Call LLM API
        response_text = self.llm_post(prompt)
        # print(f"Debug response_text: {response_text}")
        response_json = json.loads(response_text)

        # Parse answer and confidence
        answer = response_json.get("answer", "None")
        confidence = response_json.get("confidence", 0.0)
        confidence = float(confidence)
        # print(f"Debug [question]: {question} [answer]: {answer}; [confidence]: {confidence}")

        return RAGAnswer(answer, confidence, self.node_id)

    def retrieve_local_context(self, question: str) -> str:
        """Retrieve relevant context from local dataset"""
        relevant_context = self.retriever.semantic_search(self.local_dataset, question)[0]
        return str(relevant_context)

    def handle_retrieval_request(self, request: RetrievalRequest) -> RAGAnswer:
        """Handle retrieval requests from other nodes"""
        return self.generate_answer(request.question)

    def get_peer_answers(
            self,
            request: RetrievalRequest,
            peer_nodes: List[str]
    ) -> List[RAGAnswer]:
        """Get answers from peer nodes"""
        # TODO:
        # In practice, implement actual network communication
        # Here we simulate peer responses
        peer_answers = []
        for peer_id in peer_nodes:
            # Simulate peer node processing
            confidence = random.uniform(0.5, 1.0)
            answer = f"Simulated answer from peer {peer_id}"
            peer_answers.append(RAGAnswer(answer, confidence, peer_id))
        return peer_answers

    def generate_final_answer(
            self,
            question: str,
            retrieved_answers: List[RAGAnswer]
    ) -> str:
        """Generate final answer using retrieved answers as context"""
        # Construct prompt with retrieved answers
        answers_context = "\n".join(
            f"Answer from {ans.source_id} (confidence: {ans.confidence}):"
            f" {ans.text}"
            for ans in retrieved_answers
        )

        prompt = f"""Question: {question}
        Retrieved answers: {answers_context}
        Please provide a final answer based on the retrieved answers above."""

        # Call LLM API
        response_text = self.llm_post(prompt)

        return response_text


def get_dict_val(dict_item, dot_keys):
    """Get nested value from the dict"""
    keys = dot_keys.split(".")
    val = dict_item
    for key in keys:
        val = val[key]
    return val


def run_evaluation(llm_api_endpoint: str, llm_name: str, data_config: dict):
    """Run evaluation on a HuggingFace dataset"""
    # Load Huggingface dataset
    dataset = load_dataset(**data_config["load"])

    # Initialize nodes, node_0 is the global node with all datapoints
    global_node = RAGNode("node_0", llm_api_endpoint, llm_name)
    local_nodes = [
        RAGNode(f"node_{i}", llm_api_endpoint, llm_name)
        for i in range(1, 3+1)  # Create 3 nodes for demonstration
    ]

    # Distribute data among nodes
    # random pick 20 samples for testing
    for i, item in enumerate(dataset.take(20)):
        node_idx = i % len(local_nodes)
        question = get_dict_val(item, data_config["question_path"])
        answer = get_dict_val(item, data_config["answer_path"])
        datapoint = Datapoint(question, answer)

        # TODO: randomly distribute datapoint
        global_node.add_to_local_dataset(datapoint)
        local_nodes[node_idx].add_to_local_dataset(datapoint)

    # Run evaluation
    evaluation_dataset = EvaluationDataset()

    for item in global_node.local_dataset:
        # Each question is processed by a random node
        node = random.choice(local_nodes)
        peer_nodes = [n.node_id for n in local_nodes if n.node_id != node.node_id]

        # prompt_input, prediction, and expected_output
        prediction = node.process_question(item.question, peer_nodes)

        # DeepEval test case
        test_case = LLMTestCase(
            input=item.question,
            actual_output=prediction,
            expected_output=item.answer,
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
    run_evaluation(cfg.model.endpoint, cfg.model.name, cfg.data.as_dict())


if __name__ == "__main__":
    main()
