from pydantic import BaseModel


class Datapoint(BaseModel):
    topic: str
    question: str
    answer: str


class DRAGAnswer(BaseModel):
    answer: str
    relevant_knowledge: str
    relevant_score: float
    num_hops: int
    num_messages: int


class Testcase(BaseModel):
    question: str
    expected_output: str
    actual_output: str
    relevant_knowledge: str
    relevant_score: float
    num_hops: int
    num_messages: int
