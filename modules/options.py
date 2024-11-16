from jsonargparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(
        default_config_files=[
            "./config/drag.yaml", 
            "./config/model/llama32_3b.yaml", 
            "./config/dataset/trivia_qa.yaml"
        ]
    )

    parser.add_argument('--log_level', type=str, help='DEBUG, INFO, WARNING, ERROR, or CRITICAL')

    parser.add_argument("--model.base_url", type=str)
    parser.add_argument("--model.name", type=str)

    parser.add_argument("--data.load.path", type=str)
    parser.add_argument("--data.load.name", type=str)
    parser.add_argument("--data.load.split", type=str)
    parser.add_argument("--data.question_path", type=str)
    parser.add_argument("--data.answer_path", type=str)

    parser.add_argument("--drag.num_nodes", type=int)
    parser.add_argument("--drag.retrieval_confidence_threshold", type=float)
    parser.add_argument("--drag.retrieval_neighbor_num", type=int)

    parser.add_argument("--config", action="config")  

    cfg = parser.parse_args()
    return cfg
