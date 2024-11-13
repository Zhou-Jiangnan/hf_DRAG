from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCaseParams
import instructor
from openai import OpenAI
from pydantic import BaseModel


class CustomEvaluateLLM(DeepEvalBaseLLM):
    def __init__(self, llm_base_url, llm_name):
        # Structured Outputs with Ollama: https://python.useinstructor.com/hub/ollama/
        self.client = instructor.from_openai(
            OpenAI(
                base_url=llm_base_url + "/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        self.llm_name = llm_name

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        resp = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.llm_name


class Evaluator:
    def __init__(self, llm_base_url, llm_name):
        """
        DeepEval: LLM Evaluation Metrics
        https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

        :param llm_base_url: Ollama client host to connect to, http://localhost:11434 by default
        :param llm_name: Ollama LLM name
        """
        self.custom_evaluate_llm = CustomEvaluateLLM(llm_base_url=llm_base_url, llm_name=llm_name)
        self.metrics = self.load_metrics()


    def evaluate(self, evaluation_dataset):
        evaluate(evaluation_dataset, self.metrics, run_async=False, write_cache=False)

    def load_metrics(self):
        metrics = []

        # GEval: uses LLMs to evaluate LLM outputs (aka. LLM-Evals)
        # https://arxiv.org/abs/2303.16634
        # Liu, Yang, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu.
        # "G-eval: Nlg evaluation using gpt-4 with better human alignment."
        # arXiv preprint arXiv:2303.16634 (2023).
        metric = GEval(
            name="Correctness",
            criteria="Correctness - determine if the actual output is correct according to the expected output.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            model=self.custom_evaluate_llm,
        )
        metrics.append(metric)
        return metrics
