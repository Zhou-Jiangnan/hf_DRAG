from dataclasses import dataclass
import logging
import random
import requests
from typing import List, Dict

from datasets import load_dataset
from jsonargparse import ArgumentParser, CLI
from sklearn.metrics import accuracy_score, f1_score, recall_score


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


@dataclass
class LocalDatapoint:
    question: str
    context: str
    answer: str


class LocalRAGNode:
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
        self.local_dataset: List[LocalDatapoint] = []

    def add_to_local_dataset(self, datapoint: LocalDatapoint):
        """Add a new datapoint to local RAG dataset"""
        self.local_dataset.append(datapoint)

    def llm_post(self, prompt):
        body_data = {
            "model": self.llm_name,
            "prompt": prompt,
            "temperature": 0,
        }

        response = requests.post(
            self.llm_api_endpoint,
            json=body_data
        )
        response_text = response.json()["choices"][0]["text"]
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
        prompt = f"""Question: {question}
        Context: {relevant_context}
        Please provide an answer based on the context above and rate your confidence from 0.0 to 1.0 
        in the following format:
        Answer: <your_answer>
        Confidence: <confidence_score>"""

        # Call LLM API
        response_text = self.llm_post(prompt)

        # Parse answer and confidence
        answer_line = response_text.split("\n")[0]
        confidence_line = response_text.split("\n")[1]
        answer = answer_line.replace("Answer: ", "")
        confidence = float(confidence_line.replace("Confidence: ", ""))

        return RAGAnswer(answer, confidence, self.node_id)

    def retrieve_local_context(self, question: str) -> str:
        """Retrieve relevant context from local dataset"""
        # TODO:
        # Simple implementation - could be improved with better similarity search
        if not self.local_dataset:
            return ""

        # Random selection for demonstration
        # In practice, use embedding similarity or other retrieval methods
        random_datapoint = random.choice(self.local_dataset)
        return random_datapoint.context

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


class DRAGEvaluator:
    @staticmethod
    def evaluate(
            predictions: List[str],
            ground_truth: List[str]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Exact Match
        em_score = sum(p.strip() == g.strip()
                       for p, g in zip(predictions, ground_truth)) / len(predictions)

        # TODO
        # Convert to binary labels for classification metrics
        # This is a simplified approach - in practice, you might want more sophisticated
        # methods for converting text to binary labels
        binary_preds = [1 if p.strip() == g.strip() else 0
                        for p, g in zip(predictions, ground_truth)]
        binary_truth = [1] * len(ground_truth)

        metrics = {
            "exact_match": em_score,
            "accuracy": accuracy_score(binary_truth, binary_preds),
            "f1": f1_score(binary_truth, binary_preds),
            "recall": recall_score(binary_truth, binary_preds)
        }

        return metrics


def run_evaluation(llm_api_endpoint: str, llm_name: str, ds_config: dict):
    """Run evaluation on a HuggingFace dataset"""
    # Load Huggingface dataset
    dataset = load_dataset(**ds_config)

    # Initialize nodes
    nodes = [
        LocalRAGNode(f"node_{i}", llm_api_endpoint, llm_name)
        for i in range(3)  # Create 3 nodes for demonstration
    ]

    # Distribute data among nodes
    for i, item in enumerate(dataset):
        node_idx = i % len(nodes)
        datapoint = LocalDatapoint(
            item["question"],
            item.get("context", ""),  # Some datasets might not have context
            item["answer"]
        )
        nodes[node_idx].add_to_local_dataset(datapoint)

    # Run evaluation
    predictions = []
    ground_truth = []

    for item in dataset:
        # Each question is processed by a random node
        node = random.choice(nodes)
        peer_nodes = [n.node_id for n in nodes if n.node_id != node.node_id]

        prediction = node.process_question(item["question"], peer_nodes)
        predictions.append(prediction)
        ground_truth.append(item["answer"])

    # Calculate metrics
    evaluator = DRAGEvaluator()
    metrics = evaluator.evaluate(predictions, ground_truth)

    return metrics


def main():
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--model.endpoint", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model.name", default="meta-llama/Llama-3.2-1B")

    parser.add_argument("--data.path", default="mandarjoshi/trivia_qa")
    parser.add_argument("--data.name", default="rc.nocontext")
    parser.add_argument("--data.split", default="validation")

    parser.add_argument("--config", action="config")
    cfg = parser.parse_args()

    # run evaluation
    run_evaluation(cfg.model.endpoint, cfg.model.name, cfg.data.as_dict())


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
