"""
Rerank các candidate đã retrieve bằng cross-encoder (chính xác hơn dense
retrieval vì xét trực tiếp cặp (query, document) thay vì so 2 vector độc
lập, nhưng chậm hơn nên chỉ áp dụng cho top-N nhỏ sau khi đã retrieve).
"""
from sentence_transformers import CrossEncoder

_cross_encoder = None

# Model rerank nhẹ, phổ biến cho English text
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank(question: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
    """
    hits: list các dict có key 'text' (từ retrieve() trong query.py)
    Trả về list đã sắp xếp lại theo điểm cross-encoder (giảm dần),
    cắt còn top_k nếu có truyền vào.
    """
    if not hits:
        return hits

    ce = _get_cross_encoder()
    pairs = [(question, h["text"]) for h in hits]
    scores = ce.predict(pairs)

    scored_hits = list(zip(hits, scores))
    scored_hits.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for h, score in scored_hits:
        h = dict(h)  # copy, tránh mutate bản gốc
        h["rerank_score"] = float(score)
        reranked.append(h)

    if top_k is not None:
        reranked = reranked[:top_k]
    return reranked
