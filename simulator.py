import json
import random
import sys
from typing import List

from datasets import load_dataset
from jsonargparse import Namespace
from loguru import logger
import numpy as np
from tqdm import tqdm

from modules.exp_logger import ExpLogger
from modules.data_types import Datapoint, Testcase
from modules.rag_network import DRAGNetwork, CRAGNetwork
from modules.evaluator import QAEvaluator
from modules.options import parse_args


def get_dict_val(dict_item, dot_keys):
    """Get nested value from the dict"""
    keys = dot_keys.split(".")
    val = dict_item
    for key in keys:
        val = val[key]
    return val


def run_simulation(cfg: Namespace):
    # Init csv logger
    exp_logger = ExpLogger()
    config_logger = exp_logger.get_yaml_logger("config")
    metrics_logger = exp_logger.get_csv_logger("metrics")
    test_cases_logger = exp_logger.get_csv_logger("test_cases")

    # Save all config
    config_logger.log(cfg.as_dict())
    config_logger.save()

    # Load Huggingface dataset
    dataset = load_dataset(**cfg.data.load.as_dict())
    data_points: List[Datapoint] = []

    # task type:
    # - mcqa for Multiple Choice Question Answering
    # - ogqa for Open Generative Question Answering
    task_type = cfg.data.task_type

    # Only pick 20 samples from dataset for test mode
    if cfg.rag.test_mode == True:
        dataset = dataset.take(20)

    for item in dataset:
        topic = get_dict_val(item, cfg.data.topic_path)
        question = get_dict_val(item, cfg.data.question_path)
        answer = get_dict_val(item, cfg.data.answer_path)
        if task_type == "mcqa":
            choices = get_dict_val(item, cfg.data.choices_path)
            connection_term = " Select the best answer from the following candidates, replying with 1, 2, 3, or 4: "
            question = str(question) + connection_term + str(choices)
        data_point = Datapoint(topic=str(topic), question=str(question), answer=str(answer))
        data_points.append(data_point)

    # Initialize DRAG parameters
    query_confidence_threshold = cfg.rag.query_confidence_threshold
    num_query_neighbor = min(cfg.rag.num_query_neighbor, cfg.rag.num_peers - 1)
    query_ttl = cfg.rag.query_ttl

    # Initialize RAG network with peers and knowledges
    if cfg.rag.network_type == "DRAG":
        rag_net = DRAGNetwork(cfg.rag.num_peers, cfg.rag.num_peer_attachments, cfg.llm.base_url, cfg.llm.name, 
                              cfg.rag.random_seed)
    elif cfg.rag.network_type == "CRAG":
        rag_net = CRAGNetwork(cfg.llm.base_url, cfg.llm.name, cfg.rag.random_seed)
    rag_net.init_knowledge(data_points)

    # Run evaluation
    qa_evaluator = QAEvaluator()
    test_cases = []

    for idx, data_point in enumerate(tqdm(data_points, desc=f"Inferencing on {len(data_points)} test case(s)")):
        if cfg.rag.network_type == "DRAG":
            if cfg.rag.search_algorithm == "TARW":
                rag_answer = rag_net.topic_query(
                    data_point.question, 
                    num_query_neighbor=num_query_neighbor, 
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=query_ttl
                )
            elif cfg.rag.search_algorithm == "RW":
                rag_answer = rag_net.random_walk_query(
                    data_point.question,
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=query_ttl
                )
            elif cfg.rag.search_algorithm == "FL":
                rag_answer = rag_net.flooding_query(
                    data_point.question,
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=query_ttl
                )
            else:
                raise ValueError(f"Unkonw search algorithm: {cfg.rag.search_algorithm}")
        elif cfg.rag.network_type == "CRAG":
            rag_answer = rag_net.query(
                data_point.question,
                query_confidence_threshold=query_confidence_threshold
            )
        else:
            raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

        # test case
        test_case = Testcase(
            question=data_point.question,
            expected_output=data_point.answer,
            actual_output=rag_answer.answer,
            relevant_knowledge=rag_answer.relevant_knowledge,
            relevant_score=rag_answer.relevant_score,
            num_hops=rag_answer.num_hops,
            num_messages=rag_answer.num_messages,
            is_query_hit=rag_answer.is_query_hit
        )
        test_cases_logger.log(test_case.model_dump())
        test_cases.append(test_case)

        # log evaluation results regularly
        if idx % cfg.rag.log_every_n_steps == 0:
            test_cases_logger.save()
            eval_results = qa_evaluator.evaluate(test_cases)
            metrics_logger.log(eval_results)
            metrics_logger.save()

    # log final results
    test_cases_logger.save()
    eval_results = qa_evaluator.evaluate(test_cases)
    metrics_logger.log(eval_results)
    metrics_logger.save()

    logger.info(f"\nFinal Evaluation Results:\n{json.dumps(eval_results)}\n")


def main():
    # parse arguments
    cfg = parse_args()

    # Initialize random seeds
    random.seed(cfg.rag.random_seed)
    np.random.seed(cfg.rag.random_seed)

    # Changing the level of the logger
    logger.remove()  # Remove default handler.
    logger.add(sys.stderr, level=cfg.log_level)

    # run evaluation
    run_simulation(cfg)


if __name__ == "__main__":
    main()

