import json
import random
import sys
from typing import List

from datasets import load_dataset
from jsonargparse import Namespace
from loguru import logger
import numpy as np
from tqdm import tqdm

from modules.csv_logger import CSVLogger
from modules.data_types import Datapoint, Testcase
from modules.drag_network import DRAGNetwork
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
    csv_logger = CSVLogger()
    metrics_logger = csv_logger.logger("metrics")
    testcases_logger = csv_logger.logger("testcases")

    # Load Huggingface dataset
    dataset = load_dataset(**cfg.data.load.as_dict())
    data_points: List[Datapoint] = []

    # task type:
    # - mcqa for Multiple Choice Question Answering
    # - ogqa for Open Generative Question Answering
    task_type = cfg.data.task_type

    # random pick 20 samples for testing
    for _, item in enumerate(dataset.take(20)):
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
    query_confidence_threshold = cfg.drag.query_confidence_threshold
    num_query_neighbor = min(cfg.drag.num_query_neighbor, cfg.drag.num_peers - 1)
    query_ttl = cfg.drag.query_ttl

    # Initialize DRAG network with peers and knowledges
    drag_net = DRAGNetwork(cfg.drag.num_peers, cfg.drag.num_peer_attachments, cfg.llm.base_url, cfg.llm.name)
    drag_net.init_knowledge(data_points)

    # Run evaluation
    qa_evaluator = QAEvaluator()
    test_cases = []

    for idx, data_point in enumerate(tqdm(data_points, desc=f"Inferencing on {len(data_points)} test case(s)")):
        # Each question is processed by a random node
        if cfg.drag.search_algorithm == "TARW":
            drag_answer = drag_net.topic_query(
                data_point.question, 
                num_query_neighbor=num_query_neighbor, 
                query_confidence_threshold=query_confidence_threshold,
                max_ttl=query_ttl
            )
        elif cfg.drag.search_algorithm == "RW":
            drag_answer = drag_net.random_walk_query(
                data_point.question,
                query_confidence_threshold=query_confidence_threshold,
                max_ttl=query_ttl
            )
        elif cfg.drag.search_algorithm == "FL":
            drag_answer = drag_net.flooding_query(
                data_point.question,
                query_confidence_threshold=query_confidence_threshold,
                max_ttl=query_ttl
            )
        else:
            raise ValueError(f"Unkonw search algorithm: {cfg.drag.search_algorithm}")

        # test case
        test_case = Testcase(
            question=data_point.question,
            expected_output=data_point.answer,
            actual_output=drag_answer.answer,
            relevant_knowledge=drag_answer.relevant_knowledge,
            relevant_score=drag_answer.relevant_score,
            num_hops=drag_answer.num_hops,
            num_messages=drag_answer.num_messages
        )
        testcases_logger.log(test_case.model_dump())
        test_cases.append(test_case)

        # log evaluation results regularly
        if idx % cfg.drag.log_every_n_steps == 0:
            testcases_logger.save()
            eval_results = qa_evaluator.evaluate(test_cases)
            metrics_logger.log(eval_results)
            metrics_logger.save()

    # log final results
    testcases_logger.save()
    eval_results = qa_evaluator.evaluate(test_cases)
    metrics_logger.log(eval_results)
    metrics_logger.save()

    logger.info(f"\nFinal Evaluation Results:\n{json.dumps(eval_results)}\n")


def main():
    # parse arguments
    cfg = parse_args()

    # Initialize random seeds
    random.seed(cfg.drag.random_seed)
    np.random.seed(cfg.drag.random_seed)

    # Changing the level of the logger
    logger.remove()  # Remove default handler.
    logger.add(sys.stderr, level=cfg.log_level)

    # run evaluation
    run_simulation(cfg)


if __name__ == "__main__":
    main()

