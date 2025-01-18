import json
from ollama import Client


class LLM:
    def __init__(self, llm_url: str, llm_name: str):
        self.llm_name = llm_name
        self.llm_client = Client(host=llm_url)
    
    def generate(self, prompt: str):
        response = self.llm_client.generate(model=self.llm_name, prompt=prompt, format="json")
        response_text = response["response"]
        response_json = json.loads(response_text)
        return response_json
