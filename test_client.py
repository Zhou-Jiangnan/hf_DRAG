import json
import logging
import requests

logger = logging.getLogger(__file__)


def http_post(url, body_data):
    logger.info(f"[HTTP Post] {body_data}")
    response = requests.post(url, json=body_data, timeout=300)
    return response.json()


def llm_post(llm_api_endpoint, llm_name, prompt):
    body_data = {
        "model": llm_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    response = requests.post(
        llm_api_endpoint,
        json=body_data,
    )
    response_text = response.json()["response"]
    return response_text


def main():
    llm_api_endpoint = "http://localhost:11434/api/generate"
    llm_name = "llama3.2"
    prompt = """
    Why is the sky blue? 
    Please provide an answer and rate your confidence from 0.0 to 1.0. 
    Respond using JSON with keys: `answer`, `confidence`.
    """
    response_text = llm_post(llm_api_endpoint, llm_name, prompt)
    logger.info(f"result: {json.loads(response_text)}")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
