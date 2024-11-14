from typing import Dict

from pydantic import BaseModel


class RAGAnswer(BaseModel):
    text: str
    confidence: float


class Datapoint(BaseModel):
    question: str
    answer: str


class Testcase(BaseModel):
    question: str
    expected_output: str
    actual_output: str
    confidence: float
    is_retrieval_answer: bool
    retrieval_answers: Dict[int, RAGAnswer]
