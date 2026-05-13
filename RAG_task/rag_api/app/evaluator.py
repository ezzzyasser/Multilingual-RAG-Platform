import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Download NLTK tokenizer data on first run
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """BLEU score between a reference and a predicted string."""
    ref_tokens  = nltk.word_tokenize(reference.lower())
    hyp_tokens  = nltk.word_tokenize(hypothesis.lower())
    smoothing   = SmoothingFunction().method1
    try:
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    except Exception:
        score = 0.0
    return round(score, 4)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """ROUGE-1 and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_precision_at_k(retrieved_answers: list[str], relevant_answer: str, k: int = 5) -> float:
   
    if not retrieved_answers:
        return 0.0
    top_k = retrieved_answers[:k]
    hits  = sum(1 for a in top_k if relevant_answer.lower() in a.lower())
    return round(hits / len(top_k), 4)


def compute_recall_at_k(retrieved_answers: list[str], relevant_answer: str, k: int = 5) -> float:
    
    top_k = retrieved_answers[:k]
    found = any(relevant_answer.lower() in a.lower() for a in top_k)
    return 1.0 if found else 0.0


def full_evaluation(
    query: str,
    reference: str,
    predicted: str,
    retrieved_chunks: list[dict],
    k: int = 5,
) -> dict:
    """Run all metrics and return a single dict."""
    retrieved_answers = [c.get("answer", "") for c in retrieved_chunks]

    bleu   = compute_bleu(reference, predicted)
    rouge  = compute_rouge(reference, predicted)
    prec_k = compute_precision_at_k(retrieved_answers, reference, k)
    rec_k  = compute_recall_at_k(retrieved_answers, reference, k)

    return {
        "query":         query,
        "reference":     reference,
        "predicted":     predicted,
        "bleu":          bleu,
        "rouge1":        rouge["rouge1"],
        "rougeL":        rouge["rougeL"],
        "precision_at_k": prec_k,
        "recall_at_k":   rec_k,
    }
