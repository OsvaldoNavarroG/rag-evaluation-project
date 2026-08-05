from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.schemas import QueryRequest, QueryResponse
from rag.ingestion import ensure_nltk_resources
from rag.pipeline import get_default_system, run_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up all lazy resources during startup so the first request does not
    # pay the model-load and index-build cost. Each getter is cached, so this
    # runs the expensive work exactly once, here, where it is observable.
    ensure_nltk_resources()
    get_default_system()
    yield


app = FastAPI(title="RAG Evaluation API", version="1.0", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        # Serving config: hybrid + rerank is the best-performing setup in
        # the benchmark. Multi-query is disabled - it adds ~1.4s latency
        # without a measurable quality gain on the benchmark.
        result: dict[str, Any] = run_rag(
            question=request.question,
            use_hybrid=True,
            use_rerank=True,
            use_multiquery=False,
        )
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            groundedness=result["groundedness"],
            grounded_top1=result["grounded_top1"],
            faithfulness=result["faithfulness"],
            latency=result["latency"],
        )
    except Exception as e:  # noqa: BLE001 - catch-all so the endpoint returns a 500 rather than crashing; TODO: narrow and stop leaking str(e) to the client
        raise HTTPException(status_code=500, detail=str(e))
