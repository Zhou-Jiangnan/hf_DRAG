from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase


def main():
    # DeepEval: LLM Evaluation Metrics
    # https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

    # GEval: uses LLMs to evaluate LLM outputs (aka. LLM-Evals)
    # https://arxiv.org/abs/2303.16634
    # Liu, Yang, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu.
    # "G-eval: Nlg evaluation using gpt-4 with better human alignment."
    # arXiv preprint arXiv:2303.16634 (2023).
    correctness_metric = GEval(
        name="Correctness",
        criteria="Correctness - determine if the actual output is correct according to the expected output.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
    )

    test_case = LLMTestCase(
        input="The dog chased the cat up the tree, who ran up the tree?",
        # actual_output="It depends, some might consider the cat, while others might argue the dog.",
        actual_output="The cat.",
        expected_output="The cat."
    )

    correctness_metric.measure(test_case)
    print(correctness_metric.score)
    print(correctness_metric.reason)


if __name__=="__main__":
    main()
