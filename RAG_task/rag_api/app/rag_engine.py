import os
import time
import hashlib
import numpy as np
import pandas as pd
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = os.getenv("DATA_DIR", "data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K_DEFAULT   = int(os.getenv("TOP_K", 5))


# ── Singleton loader ──────────────────────────────────────────────────────────

class RAGEngine:
   
    _instance = None

    def __init__(self):
        print(" Loading RAG engine...")

        # 1. Sentence transformer (same model used in Colab)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # 2. FAISS index built in Colab
        index_path = os.path.join(DATA_DIR, "nq_faiss.index")
        self.index = faiss.read_index(index_path)
        print(f" FAISS index loaded — {self.index.ntotal} vectors")

        # 3. Chunk metadata (same nq_chunks.csv from Colab)
        chunks_path = os.path.join(DATA_DIR, "nq_chunks.csv")
        self.chunks_df = pd.read_csv(chunks_path)
        print(f" Chunks loaded — {len(self.chunks_df)} rows")

        # 4. Load pre-built embeddings for reference/verification
        embeddings_path = os.path.join(DATA_DIR, "embeddings.npy")
        self.embeddings = np.load(embeddings_path)
        print(f" Embeddings loaded — {self.embeddings.shape}")

        # 5. Groq client for answer generation
        self.groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

        # 6. In-memory answer cache
        self._cache: dict = {}

        # 7. Query embedding cache — skip re-encoding repeated queries
        self._embedding_cache: dict = {}

        print(" RAG engine ready")

    @classmethod
    def get(cls) -> "RAGEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Query helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def preprocess_query(self, query: str) -> str:
        """Lowercase, strip, normalize spaces, add question mark if missing."""
        import re
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        query = query.lower()
        if not query.endswith('?') and any(
            query.startswith(w) for w in ['who', 'what', 'where', 'when', 'why', 'how', 'which']
        ):
            query = query + '?'
        return query

    def expand_query(self, query: str) -> str:
        
        variants = [query]
        q = query.lower()

        if q.startswith('who invented'):
            variants.append(q.replace('who invented', 'who created'))
            variants.append(q.replace('who invented', 'who is the inventor of'))
        if q.startswith('what is'):
            variants.append(q.replace('what is', 'define'))
            variants.append(q.replace('what is', 'explain'))
        if q.startswith('where is'):
            variants.append(q.replace('where is', 'location of'))
        if q.startswith('when did') or q.startswith('when was'):
            variants.append(q.replace('when did', 'what year did'))
            variants.append(q.replace('when was', 'what year was'))
        if q.startswith('how many'):
            variants.append(q.replace('how many', 'what is the number of'))

        return " ".join(variants)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_DEFAULT):
        
        preprocessed = self.preprocess_query(query)
        expanded = self.expand_query(preprocessed)

        # Use cached embedding if available, otherwise encode and cache it
        if expanded in self._embedding_cache:
            embedding = self._embedding_cache[expanded]
        else:
            embedding = self.embedder.encode([expanded], normalize_embeddings=True)
            self._embedding_cache[expanded] = embedding

        scores, indices = self.index.search(np.array(embedding, dtype="float32"), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_df):
                continue
            row = self.chunks_df.iloc[idx]
            results.append({
                "text":     str(row.get("chunk", row.get("chunk_text", row.get("text", "")))),
                "question": str(row.get("question", "")),
                "answer":   str(row.get("short_answer", row.get("answer", ""))),
                "score":    float(score),
            })
        return results, expanded

    # ── Answer generation ─────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        conversation: list | None = None,
    ) -> dict:
        """
        Full RAG pipeline:
          1. Check cache
          2. Retrieve relevant chunks
          3. Build prompt with context
          4. Call Groq LLM
          5. Cache & return result
        """
        cache_key = self._cache_key(query)

        # Cache hit
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached["cache_hit"] = True
            return cached

        start = time.time()

        # Retrieval
        chunks, expanded_query = self.retrieve(query, top_k)

        # Build context string using the actual chunk text (long answer content)
        context_parts = []
        for i, c in enumerate(chunks[:5], 1):
            context_parts.append(
                f"[Source {i}]\n"
                f"Q: {c['question']}\n"
                f"A: {c['answer']}\n"
                f"Context: {c['text']}"
            )
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        # Build messages for LLM
        system_prompt = (
            "You are a helpful AI assistant. Use the provided context to answer "
            "the user's question accurately and concisely. "
            "to provide the best possible answer. "
            "Always give a direct, clear answer."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Inject multi-turn conversation history if provided
        if conversation:
            for turn in conversation[-4:]:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    messages.append(turn)

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        })

        # LLM call
        if self.groq:
            try:
                response = self.groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"LLM error: {e}"
        else:
            # Fallback: return the top chunk's stored answer
            answer = chunks[0]["answer"] if chunks else "No answer found."

        latency = round(time.time() - start, 3)
        top_score = chunks[0]["score"] if chunks else 0.0

        result = {
            "query":          query,
            "expanded_query": expanded_query,
            "answer":         answer,
            "sources":        chunks,
            "latency":        latency,
            "confidence":     top_score,
            "cache_hit":      False,
        }

        self._cache[cache_key] = result
        return result
