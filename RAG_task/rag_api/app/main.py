import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, Integer

from app.database import init_db, get_db
from app.models import QueryLog, EvaluationResult
from app.rag_engine import RAGEngine
from app.evaluator import full_evaluation


# ── Lifespan: runs once at startup ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables
    await init_db()
    # Pre-load the RAG engine (loads FAISS + model into memory)
    RAGEngine.get()
    yield  # app is now running


app = FastAPI(
    title="Multilingual RAG API",
    description="Phase 3 — Production RAG API with FAISS, Groq LLM, and SQLite logging",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: Optional[str] = None
    conversation: Optional[list] = None   # multi-turn history

class QuestionResponse(BaseModel):
    query: str
    expanded_query: str
    answer: str
    confidence: float
    latency_ms: float
    cache_hit: bool
    sources: list

class EvaluateRequest(BaseModel):
    query: str
    reference_answer: str   # ground-truth answer you want to compare against
    top_k: int = 5

class EvaluateResponse(BaseModel):
    query: str
    predicted_answer: str
    reference_answer: str
    bleu: float
    rouge1: float
    rougeL: float
    precision_at_k: float
    recall_at_k: float


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Returns system health and index stats."""
    engine = RAGEngine.get()
    return {
        "status":        "ok",
        "faiss_vectors": engine.index.ntotal,
        "chunks_loaded": len(engine.chunks_df),
        "cache_size":    len(engine._cache),
        "llm_available": engine.groq is not None,
    }


# ── POST /ask-question ────────────────────────────────────────────────────────

@app.post("/ask-question", response_model=QuestionResponse, tags=["Q&A"])
async def ask_question(
    request: QuestionRequest,
    db: AsyncSession = Depends(get_db),
):
   
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    engine = RAGEngine.get()
    result = engine.generate_answer(
        query=request.query,
        top_k=request.top_k,
        conversation=request.conversation,
    )

    latency_ms = result["latency"] * 1000

    # Persist to DB
    log = QueryLog(
        query          = request.query,
        expanded_query = result["expanded_query"],
        answer         = result["answer"],
        sources_count  = len(result["sources"]),
        latency_ms     = latency_ms,
        cache_hit      = result["cache_hit"],
        confidence     = result["confidence"],
        session_id     = request.session_id,
    )
    db.add(log)
    await db.commit()

    return QuestionResponse(
        query          = result["query"],
        expanded_query = result["expanded_query"],
        answer         = result["answer"],
        confidence     = result["confidence"],
        latency_ms     = round(latency_ms, 2),
        cache_hit      = result["cache_hit"],
        sources        = result["sources"][:3],   # return top 3 sources
    )


# ── POST /evaluate ────────────────────────────────────────────────────────────

@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate(
    request: EvaluateRequest,
    db: AsyncSession = Depends(get_db),
):
    
    if not request.query.strip() or not request.reference_answer.strip():
        raise HTTPException(status_code=400, detail="Query and reference_answer are required.")

    engine = RAGEngine.get()
    result = engine.generate_answer(query=request.query, top_k=request.top_k)

    metrics = full_evaluation(
        query           = request.query,
        reference       = request.reference_answer,
        predicted       = result["answer"],
        retrieved_chunks= result["sources"],
        k               = request.top_k,
    )

    # Persist evaluation result
    eval_row = EvaluationResult(
        query           = request.query,
        reference       = request.reference_answer,
        predicted       = result["answer"],
        bleu_score      = metrics["bleu"],
        rouge1_score    = metrics["rouge1"],
        rougeL_score    = metrics["rougeL"],
        precision_at_k  = metrics["precision_at_k"],
        recall_at_k     = metrics["recall_at_k"],
    )
    db.add(eval_row)
    await db.commit()

    return EvaluateResponse(
        query            = request.query,
        predicted_answer = result["answer"],
        reference_answer = request.reference_answer,
        bleu             = metrics["bleu"],
        rouge1           = metrics["rouge1"],
        rougeL           = metrics["rougeL"],
        precision_at_k   = metrics["precision_at_k"],
        recall_at_k      = metrics["recall_at_k"],
    )


# ── GET /analytics ────────────────────────────────────────────────────────────

@app.get("/analytics", tags=["System"])
async def analytics(db: AsyncSession = Depends(get_db)):
    """Returns aggregated query statistics from the database."""
    result = await db.execute(
        select(
            func.count(QueryLog.id).label("total_queries"),
            func.avg(QueryLog.latency_ms).label("avg_latency_ms"),
            func.sum(QueryLog.cache_hit.cast(Integer)).label("cache_hits"),
            func.avg(QueryLog.confidence).label("avg_confidence"),
        )
    )
    row = result.fetchone()

    return {
        "total_queries":   row.total_queries or 0,
        "avg_latency_ms":  round(row.avg_latency_ms or 0, 2),
        "cache_hits":      int(row.cache_hits or 0),
        "avg_confidence":  round(row.avg_confidence or 0, 4),
    }
