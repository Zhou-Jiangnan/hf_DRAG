from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


class Evaluator:
    # DeepEval: LLM Evaluation Metrics
    # https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

    def __init__(self):
        self.metrics = self.load_metrics()

    def evaluate(self, evaluation_dataset):
        evaluate(evaluation_dataset, self.metrics)

    @staticmethod
    def load_metrics():
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
        )
        metrics.append(metric)
        return metrics
