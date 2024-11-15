import json
import random
import sys
from typing import List, Dict, Optional

from datasets import load_dataset
from jsonargparse import Namespace
from loguru import logger
from ollama import Client
from tqdm import tqdm

from modules.csv_logger import CSVLogger
from modules.data_types import Datapoint, RAGAnswer, Testcase
from modules.evaluator import AdvancedQAEvaluator
from modules.options import parse_args
from modules.retriever import ContextRetriever


class RAGNode:
    def __init__(
            self,
            node_id: int,
            llm_base_url: str,
            llm_name: str,
    ):
        self.node_id = node_id
        self.llm_name = llm_name
        self.llm_client = Client(host=llm_base_url)
        self.retriever = ContextRetriever()
        self.local_dataset: List[Datapoint] = []

    def add_to_local_dataset(self, datapoint: Datapoint):
        """Add a new datapoint to local RAG dataset"""
        self.local_dataset.append(datapoint)
    
    def invoke_llm(self, prompt: str) -> RAGAnswer:
        response = self.llm_client.generate(model=self.llm_name, prompt=prompt, format="json")
        response_text = response["response"]

        # logger.debug(f"response_text: {response_text}")
        response_json = json.loads(response_text)

        # Parse answer and confidence
        answer = response_json.get("answer", "None")
        answer = str(answer)
        confidence = response_json.get("confidence", 0.0)
        confidence = float(confidence)
        # logger.debug(f"[question]: {question} [answer]: {answer}; [confidence]: {confidence}")
        return RAGAnswer(text=answer, confidence=confidence)

    def generate_answer(self, question: str, retrieved_answers: Dict[int, RAGAnswer]|None = None) -> RAGAnswer:
        """Generate answer using local dataset and LLM"""
        # Find most relevant info from local dataset
        retrieved_info = self.retrieve_local_info(question)

        # Construct prompt with retrieved answers
        if retrieved_answers is not None:
            global_info = "\n".join(
                f"Answer from {node_id} (confidence: {ans.confidence}):"
                f" {ans.text}"
                for node_id, ans in retrieved_answers.items()
            )
            retrieved_info = retrieved_info + "\n" + global_info

        # Construct prompt with retrieved information
        prompt = f"""
        Question:
        {question}

        Retrieved Information:
        {retrieved_info}

        Please provide an answer using retrieved information, along with a confidence score between 0.0 and 1.0. 
        Assign a confidence of 1.0 only if the retrieved information is directly relevant, accurate, 
        and fully supports the answer. For responses relying on partial or inferred information, 
        use confidence levels below 1.0.

        Respond using JSON with keys: `answer`, `confidence`."""

        # Execute LLM
        rag_answer = self.invoke_llm(prompt)
        return rag_answer

    def retrieve_local_info(self, question: str) -> str:
        """Retrieve relevant information from local dataset"""
        local_sentences = [f"[Question]:{datapoint.question}[Answer]:{datapoint.answer}"
                           for datapoint in self.local_dataset]
        relevant_info = self.retriever.semantic_search(local_sentences, question)[0]
        return str(relevant_info)


class DistributedRAGSystem:
    def __init__(
            self, 
            llm_base_url: str,
            llm_name: str, 
            num_nodes: int, 
            datapoints: List[Datapoint], 
            retrieval_confidence_threshold: float,
            retrieval_neighbor_num: int,
        ):

        self.retrieval_confidence_threshold = retrieval_confidence_threshold
        self.retrieval_neighbor_num = retrieval_neighbor_num
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

    
    def query(
            self, datapoint: Datapoint, selected_node_id: Optional[int] = None
    ) -> (RAGAnswer, bool, Dict[int, RAGAnswer]):
        """
        Query a specific question on a certain node in the cluster.

        Args:
            datapoint: Datapoint, the question and the user-provided correct answer, to update node's local dataset
            selected_node_id: Optional[int]: which node to ask question
        Returns:
            Tuple(RAGAnswer, bool, Dict[int, RAGAnswer]): final answer, is_retrieval_answer, and retrieval_answers
        """
        if selected_node_id is None:
            selected_node_id = random.choice(list(self.nodes.keys()))
        
        selected_node = self.nodes[selected_node_id]

        # Get answer from starting node
        original_answer = selected_node.generate_answer(datapoint.question)

        # If original answer is confident enough, return
        if original_answer.confidence > self.retrieval_confidence_threshold:
            logger.debug(f"confident enough for question: {datapoint.question}, return: {original_answer}")
            return original_answer, False, {}

        # Select random subset of neighbors to retrieval
        selected_neighbor_ids = random.sample(list(self.nodes.keys()), self.retrieval_neighbor_num)
        logger.debug(f"not confident for question: {datapoint.question}, "
                     f"fetch answers from neighbors {selected_neighbor_ids}")
        
        retrieval_answers: Dict[int, RAGAnswer] = {}
        
        # Get answers from selected neighbors
        for neighbor_id in selected_neighbor_ids:
            neighbor = self.nodes[neighbor_id]
            retrieval_answers[neighbor_id] = neighbor.generate_answer(datapoint.question)

        # TODO: Rank and cache answers by confidence and select top k
        logger.debug(f"fetched answers from neighbors {selected_neighbor_ids}:\n {retrieval_answers}")
        final_answer = selected_node.generate_answer(datapoint.question, retrieval_answers)
        logger.debug(f"after aggregating answers from neighbors, final answer: {final_answer}")

        selected_node.add_to_local_dataset(Datapoint(question=datapoint.question, answer=datapoint.answer))
        
        return final_answer, True, retrieval_answers


def get_dict_val(dict_item, dot_keys):
    """Get nested value from the dict"""
    keys = dot_keys.split(".")
    val = dict_item
    for key in keys:
        val = val[key]
    return val


def run_simulation(model_cfg: Namespace, data_cfg: Namespace, drag_cfg: Namespace):
    """Run evaluation on a HuggingFace dataset"""
    # Init csv logger
    csv_logger = CSVLogger()
    metrics_logger = csv_logger.logger("metrics")
    testcases_logger = csv_logger.logger("testcases")

    # Load Huggingface dataset
    dataset = load_dataset(**data_cfg.load.as_dict())
    datapoints: List[Datapoint] = []

    # random pick 20 samples for testing
    for i, item in enumerate(dataset.take(20)):
        question = get_dict_val(item, data_cfg.question_path)
        answer = get_dict_val(item, data_cfg.answer_path)
        datapoint = Datapoint(question=str(question), answer=str(answer))
        datapoints.append(datapoint)

    # Initialize distributed RAG system
    drag_system = DistributedRAGSystem(
        llm_base_url=model_cfg.base_url,
        llm_name=model_cfg.name,
        num_nodes=drag_cfg.num_nodes,
        datapoints=datapoints,
        retrieval_confidence_threshold=drag_cfg.retrieval_confidence_threshold,
        retrieval_neighbor_num=drag_cfg.retrieval_neighbor_num,
    )

    # Run evaluation
    test_cases = []

    for datapoint in tqdm(datapoints, desc=f"Inferencing on {len(datapoints)} test case(s)"):
        # Each question is processed by a random node
        drag_answer, is_retrieval_answer, retrieval_answers = drag_system.query(datapoint)

        # test case
        test_case = Testcase(
            question=datapoint.question,
            expected_output=datapoint.answer,
            actual_output=drag_answer.text,
            confidence=drag_answer.confidence,
            is_retrieval_answer=is_retrieval_answer,
            retrieval_answers=retrieval_answers,
        )
        testcases_logger.log(test_case.model_dump())
        testcases_logger.save()
        test_cases.append(test_case)

    # Calculate metrics
    qa_evaluator = AdvancedQAEvaluator()
    results = qa_evaluator.evaluate(test_cases)

    # log results
    metrics_logger.log(results)
    metrics_logger.save()
    logger.info(f"Evaluation Results:\n{json.dumps(results)}")


def main():
    # parse arguments
    cfg = parse_args()

    # Changing the level of the logger
    logger.remove()  # Remove default handler.
    logger.add(sys.stderr, level=cfg.log_level)

    # run evaluation
    run_simulation(cfg.model, cfg.data, cfg.drag)


if __name__ == "__main__":
    main()
