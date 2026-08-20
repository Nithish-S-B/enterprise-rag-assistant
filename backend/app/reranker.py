"""Local cross-encoder reranking for dense-retrieval candidates."""
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Load the model once per process so every query can reuse it.
_model = CrossEncoder(MODEL_NAME)


def rerank(query: str, candidates: list[dict], final_k: int = 4) -> list[dict]:
    """Score dense-retrieval candidates and return the top final_k records.

    Each returned record retains the original retrieval fields (id, text,
    metadata, and distance) and adds a reranker_score field.
    """
    if final_k < 1:
        raise ValueError("final_k must be at least 1.")
    if not candidates:
        return []

    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = _model.predict(pairs)

    scored_candidates = []
    for candidate, score in zip(candidates, scores):
        scored_candidate = dict(candidate)
        scored_candidate["reranker_score"] = float(score)
        scored_candidates.append(scored_candidate)

    scored_candidates.sort(key=lambda candidate: candidate["reranker_score"], reverse=True)
    return scored_candidates[:final_k]
