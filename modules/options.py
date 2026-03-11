from jsonargparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(
        default_config_files=[
            "./config/rag.yaml", 
            "./config/llm/llama32_3b.yaml", 
            "./config/data/mmlu.yaml"
        ]
    )

    parser.add_argument('--log_level', type=str, help='DEBUG, INFO, WARNING, ERROR, or CRITICAL')

    parser.add_argument("--llm.base_url", type=str)
    parser.add_argument("--llm.name", type=str)
    parser.add_argument("--llm.num_ctx", type=int)

    parser.add_argument("--data.load.path", type=str)
    parser.add_argument("--data.load.name", type=str)
    parser.add_argument("--data.load.split", type=str)
    parser.add_argument("--data.task_type", type=str)
    parser.add_argument("--data.topic_path", type=str)
    parser.add_argument("--data.question_path", type=str)
    parser.add_argument("--data.choices_path", type=str)
    parser.add_argument("--data.answer_path", type=str)
    parser.add_argument("--data.num_samples", type=int)

    parser.add_argument("--rag.random_seed", type=int)
    parser.add_argument("--rag.log_every_n_steps", type=int)
    parser.add_argument("--rag.test_mode", type=bool)
    parser.add_argument("--rag.test_num_samples", type=int)

    parser.add_argument("--rag.network_type", type=str)
    parser.add_argument("--rag.num_peers", type=int)
    parser.add_argument("--rag.num_peer_attachments", type=int)
    parser.add_argument("--rag.search_algorithm", type=str)
    parser.add_argument("--rag.query_confidence_threshold", type=float)
    parser.add_argument("--rag.num_query_neighbor", type=int)
    parser.add_argument("--rag.query_ttl", type=int)
    parser.add_argument("--rag.filter_out_topic_ratio", type=float)
    parser.add_argument("--rag.filter_out_qa_ratio", type=float)

    parser.add_argument("--rag.ppo_train_episodes", type=int)
    parser.add_argument("--rag.ppo_update_epochs", type=int)
    parser.add_argument("--rag.ppo_learning_rate", type=float)
    parser.add_argument("--rag.ppo_clip_epsilon", type=float)
    parser.add_argument("--rag.ppo_entropy_coef", type=float)
    parser.add_argument("--rag.ppo_value_coef", type=float)
    parser.add_argument("--rag.ppo_gamma", type=float)
    parser.add_argument("--rag.ppo_gae_lambda", type=float)
    parser.add_argument("--rag.ppo_reward_hit", type=float)
    parser.add_argument("--rag.ppo_reward_miss", type=float)
    parser.add_argument("--rag.ppo_message_penalty", type=float)
    parser.add_argument("--rag.ppo_hop_penalty", type=float)
    parser.add_argument("--rag.ppo_relevance_weight", type=float)
    parser.add_argument("--rag.ppo_hidden_dim", type=int)
    parser.add_argument("--rag.ppo_max_candidates", type=int)

    parser.add_argument("--config", action="config")

    cfg = parser.parse_args()
    return cfg
