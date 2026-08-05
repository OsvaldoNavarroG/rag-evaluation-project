
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[int]
    groundedness: bool | None = None
    grounded_top1: bool | None = None
    faithfulness: bool | None = None
    latency: dict[str, float] | None = None