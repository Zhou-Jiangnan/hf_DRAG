import json
import logging
import requests

logger = logging.getLogger(__file__)


def http_post(url, body_data):
    logger.info(f"[HTTP Post] {body_data}")
    response = requests.post(url, json=body_data, timeout=300)
    return response.json()


def main():
    url = "http://localhost:8000/v1/completions"
    body_data = {
        "model": "meta-llama/Llama-3.2-1B",
        "prompt": "San Francisco is a",
        "temperature": 0,
    }
    result = http_post(url, body_data)
    logger.info(f"result: {json.dumps(result)}")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
