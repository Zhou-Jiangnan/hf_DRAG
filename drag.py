import json
import logging
import random
from typing import List, Dict, Optional

from datasets import load_dataset
from jsonargparse import ArgumentParser
from ollama import Client
from pydantic import BaseModel
from tqdm import tqdm

from model.evaluator import AdvancedQAEvaluator
from model.retriever import ContextRetriever

logger = logging.getLogger(__file__)


class RAGAnswer(BaseModel):
    text: str
    confidence: float


class Datapoint(BaseModel):
    question: str
    answer: str


class Testcase(BaseModel):
    actual_output: str
    expected_output: str


class RAGNode:
    def __init__(
            self,
            node_id: int,
            llm_base_url: str,
            llm_name: str,
            confidence_threshold: float = 0.7,
            num_retrieval_answers: int = 3
    ):
        self.node_id = node_id
        self.llm_name = llm_name
        self.llm_client = Client(host=llm_base_url)
        self.confidence_threshold = confidence_threshold
        self.num_retrieval_answers = num_retrieval_answers
        self.retriever = ContextRetriever()
        self.local_dataset: List[Datapoint] = []

    def add_to_local_dataset(self, datapoint: Datapoint):
        """Add a new datapoint to local RAG dataset"""
        self.local_dataset.append(datapoint)
    
    def execute_llm(self, prompt: str) -> RAGAnswer:
        response = self.llm_client.generate(model=self.llm_name, prompt=prompt, format="json")
        response_text = response["response"]

        # print(f"Debug response_text: {response_text}")
        response_json = json.loads(response_text)

        # Parse answer and confidence
        answer = response_json.get("answer", "None")
        answer = str(answer)
        confidence = response_json.get("confidence", 0.0)
        confidence = float(confidence)
        # print(f"Debug [question]: {question} [answer]: {answer}; [confidence]: {confidence}")
        return RAGAnswer(text=answer, confidence=confidence)

    def generate_answer(self, question: str, retrieved_answers: Dict[int, RAGAnswer]|None = None) -> RAGAnswer:
        """Generate answer using local dataset and LLM"""
        # Find most relevant context from local dataset
        relevant_context = self.retrieve_local_context(question)

        # Construct prompt with retrieved answers
        if retrieved_answers is not None:
            retrieved_context = "\n".join(
                f"Answer from {node_id} (confidence: {ans.confidence}):"
                f" {ans.text}"
                for node_id, ans in retrieved_answers.items()
            )
            relevant_context = relevant_context + "\n" + retrieved_context
            # print(f"Debug relevant_context: {relevant_context}")

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
        local_sentences = [f"[Question]:{datapoint.question}[Answer]:{datapoint.answer}"
                           for datapoint in self.local_dataset]
        relevant_context = self.retriever.semantic_search(local_sentences, question)[0]
        return str(relevant_context)


class DistributedRAGSystem:
    def __init__(
            self, 
            llm_base_url: str,
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
                llm_base_url=llm_base_url,
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


def run_simulation(llm_base_url: str, llm_name: str, data_config: dict, num_nodes: int = 10):
    """Run evaluation on a HuggingFace dataset"""
    # Load Huggingface dataset
    dataset = load_dataset(**data_config["load"])
    datapoints: List[Datapoint] = []

    # random pick 20 samples for testing
    for i, item in enumerate(dataset.take(20)):
        question = get_dict_val(item, data_config["question_path"])
        answer = get_dict_val(item, data_config["answer_path"])
        datapoint = Datapoint(question=str(question), answer=str(answer))
        datapoints.append(datapoint)

    # Initialize distributed RAG system
    drag_system = DistributedRAGSystem(
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        num_nodes=num_nodes,
        datapoints=datapoints,
        confidence_threshold=0.9,
    )

    # Run evaluation
    test_cases = []

    for datapoint in tqdm(datapoints, desc=f"Inferencing on {len(datapoints)} test case(s)"):
        # Each question is processed by a random node
        drag_answer = drag_system.query(datapoint.question)

        # DeepEval test case
        test_case = Testcase(
            actual_output=drag_answer.text,
            expected_output=datapoint.answer,
        )
        # print(f"Debug test_case: {test_case}")
        test_cases.append(test_case)

    # Calculate metrics
    # qa_evaluator = QAEvaluator()
    qa_evaluator = AdvancedQAEvaluator()
    results = qa_evaluator.evaluate(test_cases)

    # Print results
    print("\nEvaluation Results:")
    print("-" * 20)
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}%")


def main():
    # parse arguments
    parser = ArgumentParser(default_config_files=["./config/default.yaml"])
    parser.add_argument("--model.base_url", type=str)
    parser.add_argument("--model.name", type=str)

    parser.add_argument("--data.load.path", type=str)
    parser.add_argument("--data.load.name", type=str)
    parser.add_argument("--data.load.split", type=str)
    parser.add_argument("--data.question_path", type=str)
    parser.add_argument("--data.answer_path", type=str)

    cfg = parser.parse_args()

    # run evaluation
    run_simulation(cfg.model.base_url, cfg.model.name, cfg.data.as_dict())


if __name__ == "__main__":
    main()
