from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from app.database import Base


class QueryLog(Base):
    """Every question asked through the API."""
    __tablename__ = "query_logs"

    id            = Column(Integer, primary_key=True, index=True)
    query         = Column(Text, nullable=False)
    expanded_query = Column(Text, nullable=True)          # after query expansion
    answer        = Column(Text, nullable=True)
    sources_count = Column(Integer, default=0)
    latency_ms    = Column(Float, default=0.0)
    cache_hit     = Column(Boolean, default=False)
    confidence    = Column(Float, default=0.0)            # top similarity score
    session_id    = Column(String(64), nullable=True)     # for multi-turn tracking
    created_at    = Column(DateTime, default=datetime.utcnow)


class EvaluationResult(Base):
    """Results from the /evaluate endpoint."""
    __tablename__ = "evaluation_results"

    id            = Column(Integer, primary_key=True, index=True)
    query         = Column(Text, nullable=False)
    reference     = Column(Text, nullable=False)          # ground-truth answer
    predicted     = Column(Text, nullable=False)          # model answer
    bleu_score    = Column(Float, default=0.0)
    rouge1_score  = Column(Float, default=0.0)
    rougeL_score  = Column(Float, default=0.0)
    precision_at_k = Column(Float, default=0.0)
    recall_at_k   = Column(Float, default=0.0)
    created_at    = Column(DateTime, default=datetime.utcnow)
