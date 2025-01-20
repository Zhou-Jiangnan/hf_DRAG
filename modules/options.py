from jsonargparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(
        default_config_files=[
            "./config/drag.yaml", 
            "./config/llm/llama32_3b.yaml", 
            "./config/data/mmlu.yaml"
        ]
    )

    parser.add_argument('--log_level', type=str, help='DEBUG, INFO, WARNING, ERROR, or CRITICAL')

    parser.add_argument("--llm.base_url", type=str)
    parser.add_argument("--llm.name", type=str)

    parser.add_argument("--data.load.path", type=str)
    parser.add_argument("--data.load.name", type=str)
    parser.add_argument("--data.load.split", type=str)
    parser.add_argument("--data.task_type", type=str)
    parser.add_argument("--data.topic_path", type=str)
    parser.add_argument("--data.question_path", type=str)
    parser.add_argument("--data.choices_path", type=str)
    parser.add_argument("--data.answer_path", type=str)

    parser.add_argument("--drag.random_seed", type=int)
    parser.add_argument("--drag.log_every_n_steps", type=int)
    parser.add_argument("--drag.num_peers", type=int)
    parser.add_argument("--drag.num_peer_attachments", type=int)
    parser.add_argument("--drag.search_algorithm", type=str)
    parser.add_argument("--drag.query_confidence_threshold", type=float)
    parser.add_argument("--drag.num_query_neighbor", type=int)
    parser.add_argument("--drag.query_ttl", type=int)

    parser.add_argument("--config", action="config")

    cfg = parser.parse_args()
    return cfg

