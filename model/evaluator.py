from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCaseParams
from ollama import Client
from outlines import generate, models


class CustomEvaluateLLM(DeepEvalBaseLLM):
    def __init__(self, llm_base_url, llm_name):
        # TODO: integrate outlines
        # https://github.com/dottxt-ai/outlines/pull/1142
        # https://github.com/dottxt-ai/outlines/issues/1135
        # models.openai()
        self.llm_client = Client(host=llm_base_url)
        self.llm_name = llm_name

    def load_model(self):
        return self.llm_client

    def generate(self, prompt: str) -> str:
        response = self.llm_client.generate(model=self.llm_name, prompt=prompt)
        return response["response"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

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
        evaluate(evaluation_dataset, self.metrics, write_cache=False)

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
