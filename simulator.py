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
from modules.ppo_router import PPORouter, PPOConfig
from modules.grpo_router import GRPORouter, GRPOConfig
from modules.rag_network import DRAGNetwork, CRAGNetwork, NoRAGNetwork
from modules.evaluator import QAEvaluator
from modules.options import parse_args


def get_nested_value(data_dict: dict, dot_key_path: str):
    """
    Retrieves a nested value from a dictionary using a dot-separated key path.

    Special syntax:
    - "__const__:VALUE": returns VALUE directly (useful for datasets without topic labels).

    Args:
        data_dict: The dictionary to retrieve the value from.
        dot_key_path: A string representing the nested keys separated by dots (e.g., "key1.key2.key3").

    Returns:
        The value at the specified path in the dictionary.
    """
    if dot_key_path.startswith("__const__:"):
        return dot_key_path.split(":", 1)[1]

    keys = dot_key_path.split(".")
    value = data_dict
    for key in keys:
        value = value[key]
    return value


def normalize_field_value(value, *, prefer_first: bool = False) -> str:
    """Normalize dataset field values into strings for Datapoint construction."""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        if prefer_first:
            return str(value[0])
        return " ".join(str(v) for v in value)
    return str(value)


def run_simulation(cfg: Namespace):
    # Init csv logger
    exp_logger = ExpLogger()
    logger.info(f"Experiment Log Directory: {exp_logger.experiment_dir}")
    config_logger = exp_logger.get_yaml_logger("config")
    metrics_logger = exp_logger.get_csv_logger("metrics")
    test_cases_logger = exp_logger.get_csv_logger("test_cases")

    # Save all config
    config_logger.log(cfg.as_dict())
    config_logger.save()

    # Load Huggingface dataset
    dataset = load_dataset(**cfg.data.load.as_dict())
    data_points: List[Datapoint] = []
    all_topics = set()

    # task type:
    # - mcqa for Multiple Choice Question Answering
    # - ogqa for Open Generative Question Answering
    task_type = cfg.data.task_type

    if cfg.rag.test_mode:
        # Only pick configurable number of samples from dataset in test mode
        dataset = dataset.select(range(min(cfg.rag.test_num_samples, len(dataset))))
    else:
        if cfg.data.num_samples is not None:
            # Sample data if num_samples is specified
            dataset = dataset.shuffle(seed=cfg.rag.random_seed).take(min(cfg.data.num_samples, len(dataset)))
        else:
            dataset = dataset.shuffle(seed=cfg.rag.random_seed)

    # Prepare data points
    for item in dataset:
        topic = normalize_field_value(get_nested_value(item, cfg.data.topic_path))
        question = normalize_field_value(get_nested_value(item, cfg.data.question_path))
        answer = normalize_field_value(get_nested_value(item, cfg.data.answer_path), prefer_first=True)
        if task_type == "mcqa":
            choices = normalize_field_value(get_nested_value(item, cfg.data.choices_path))
            connection_term = " Select the best answer from the following candidates, replying with 1, 2, 3, or 4: "
            question = question + connection_term + choices

        data_point = Datapoint(topic=topic, question=question, answer=answer)
        all_topics.add(str(topic))
        data_points.append(data_point)
    
    if cfg.rag.network_type == "DRAG":
        filtered_data_points = data_points
    elif cfg.rag.network_type == "CRAG":
        # Filter out a portion of data points in CRAG for comparison
        num_topics_to_keep = int(len(all_topics) * (1.0 - cfg.rag.filter_out_topic_ratio))
        filtered_topics = random.sample(list(all_topics), k=num_topics_to_keep)
        filtered_data_points = [
            dp for dp in data_points if dp.topic in filtered_topics
        ]
        num_datapoints_to_keep = int(len(filtered_data_points) * (1.0 - cfg.rag.filter_out_qa_ratio))
        filtered_data_points = random.sample(filtered_data_points, k=num_datapoints_to_keep)
    elif cfg.rag.network_type == "NoRAG":
        filtered_data_points = []
    else:
        raise ValueError(f"Unknown network type: {cfg.rag.network_type}")

    # Initialize DRAG parameters
    query_confidence_threshold = cfg.rag.query_confidence_threshold
    num_query_neighbor = min(cfg.rag.num_query_neighbor, cfg.rag.num_peers - 1)
    query_ttl = cfg.rag.query_ttl

    # Initialize RAG network with peers and knowledges
    ppo_router = None
    grpo_router = None
    if cfg.rag.network_type == "DRAG":
        rag_net = DRAGNetwork(cfg.rag.num_peers, cfg.rag.num_peer_attachments, cfg.llm.base_url, cfg.llm.name, 
                              cfg.llm.num_ctx, cfg.rag.random_seed)
    elif cfg.rag.network_type == "CRAG":
        rag_net = CRAGNetwork(cfg.llm.base_url, cfg.llm.name, cfg.llm.num_ctx, cfg.rag.random_seed)
    elif cfg.rag.network_type == "NoRAG":
        rag_net = NoRAGNetwork(cfg.llm.base_url, cfg.llm.name, cfg.llm.num_ctx, cfg.rag.random_seed)
    else:
        raise ValueError(f"Unknown network type: {cfg.rag.network_type}")
    rag_net.init_knowledge(filtered_data_points)

    if cfg.rag.network_type == "DRAG" and cfg.rag.search_algorithm == "PPO":
        ppo_cfg = PPOConfig(
            hidden_dim=cfg.rag.ppo_hidden_dim,
            clip_epsilon=cfg.rag.ppo_clip_epsilon,
            value_coef=cfg.rag.ppo_value_coef,
            entropy_coef=cfg.rag.ppo_entropy_coef,
            learning_rate=cfg.rag.ppo_learning_rate,
            gamma=cfg.rag.ppo_gamma,
            gae_lambda=cfg.rag.ppo_gae_lambda,
            update_epochs=cfg.rag.ppo_update_epochs,
            max_candidates=cfg.rag.ppo_max_candidates,
        )
        ppo_router = PPORouter(ppo_cfg, device=cfg.rag.ppo_device)
        logger.info(f"PPO router device: {ppo_router.device}")
        rag_net.ppo_train(
            router=ppo_router,
            data_points=filtered_data_points,
            num_episodes=cfg.rag.ppo_train_episodes,
            max_ttl=query_ttl,
            query_confidence_threshold=query_confidence_threshold,
            reward_hit=cfg.rag.ppo_reward_hit,
            reward_miss=cfg.rag.ppo_reward_miss,
            message_penalty=cfg.rag.ppo_message_penalty,
            hop_penalty=cfg.rag.ppo_hop_penalty,
            relevance_weight=cfg.rag.ppo_relevance_weight,
            progress_weight=cfg.rag.ppo_progress_weight,
            topic_match_bonus=cfg.rag.ppo_topic_match_bonus,
            revisit_penalty=cfg.rag.ppo_revisit_penalty,
            hop_progressive_penalty=cfg.rag.ppo_hop_progressive_penalty,
            early_hit_bonus=cfg.rag.ppo_early_hit_bonus,
        )
    elif cfg.rag.network_type == "DRAG" and cfg.rag.search_algorithm == "GRPO":
        grpo_cfg = GRPOConfig(
            hidden_dim=cfg.rag.grpo_hidden_dim,
            clip_epsilon=cfg.rag.grpo_clip_epsilon,
            value_coef=cfg.rag.grpo_value_coef,
            entropy_coef=cfg.rag.grpo_entropy_coef,
            learning_rate=cfg.rag.grpo_learning_rate,
            gamma=cfg.rag.grpo_gamma,
            update_epochs=cfg.rag.grpo_update_epochs,
            max_candidates=cfg.rag.grpo_max_candidates,
            group_size=cfg.rag.grpo_group_size,
        )
        grpo_router = GRPORouter(grpo_cfg, device=cfg.rag.grpo_device)
        logger.info(f"GRPO router device: {grpo_router.device}")
        rag_net.grpo_train(
            router=grpo_router,
            data_points=filtered_data_points,
            num_episodes=cfg.rag.grpo_train_episodes,
            max_ttl=query_ttl,
            query_confidence_threshold=query_confidence_threshold,
            reward_hit=cfg.rag.ppo_reward_hit,
            reward_miss=cfg.rag.ppo_reward_miss,
            message_penalty=cfg.rag.ppo_message_penalty,
            hop_penalty=cfg.rag.ppo_hop_penalty,
            relevance_weight=cfg.rag.ppo_relevance_weight,
            progress_weight=cfg.rag.ppo_progress_weight,
            topic_match_bonus=cfg.rag.ppo_topic_match_bonus,
            revisit_penalty=cfg.rag.ppo_revisit_penalty,
            hop_progressive_penalty=cfg.rag.ppo_hop_progressive_penalty,
            early_hit_bonus=cfg.rag.ppo_early_hit_bonus,
        )

    # Run evaluation
    qa_evaluator = QAEvaluator()

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
            elif cfg.rag.search_algorithm == "PPO":
                rag_answer = rag_net.ppo_query(
                    data_point.question,
                    router=ppo_router,
                    query_confidence_threshold=query_confidence_threshold,
                    max_ttl=query_ttl
                )
            elif cfg.rag.search_algorithm == "GRPO":
                rag_answer = rag_net.grpo_query(
                    data_point.question,
                    router=grpo_router,
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
        elif cfg.rag.network_type == "NoRAG":
            rag_answer = rag_net.query(
                data_point.question
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
        qa_evaluator.add(test_case)

        # log evaluation results regularly
        if idx % cfg.rag.log_every_n_steps == 0:
            test_cases_logger.save()
            eval_results = qa_evaluator.get_results()
            metrics_logger.log(eval_results)
            metrics_logger.save()

    # log final results
    test_cases_logger.save()
    eval_results = qa_evaluator.get_results()
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
