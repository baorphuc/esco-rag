"""
Bước 3: Retrieval + Generation (RAG end-to-end).

Chạy: python query.py "Kỹ năng nào cần thiết cho technical director?"

Cần biến môi trường GEMINI_API_KEY (hoặc set trong config.py).
"""
import sys
import json

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_POOL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

from rerank import rerank

_embed_model = None
_collection = None
_genai_client = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("Chưa set GEMINI_API_KEY (env var hoặc trong config.py)")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def retrieve(question: str, k: int = TOP_K_RETRIEVAL):
    model = _get_embed_model()
    collection = _get_collection()

    query_emb = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits


def build_prompt(question: str, hits: list[dict]) -> str:
    context_blocks = []
    for i, h in enumerate(hits, 1):
        context_blocks.append(f"[Source {i}: {h['title']}]\n{h['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Bạn là trợ lý trả lời câu hỏi về nghề nghiệp và kỹ năng dựa trên dữ liệu ESCO.
Chỉ dùng thông tin trong phần CONTEXT bên dưới. Nếu không có đủ thông tin, nói rõ là không tìm thấy.
Khi trả lời, trích dẫn nguồn bằng [Source N].

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def generate_answer(question: str, hits: list[dict]) -> str:
    client = _get_genai_client()
    prompt = build_prompt(question, hits)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def answer(question: str, k: int = TOP_K_RETRIEVAL, verbose: bool = True, use_rerank: bool = True):
    hits = retrieve(question, k=TOP_K_POOL if use_rerank else k)
    if use_rerank:
        hits = rerank(question, hits, top_k=k)
    if verbose:
        print("\n--- Retrieved ---")
        for h in hits:
            score = f" rerank={h['rerank_score']:.3f}" if "rerank_score" in h else ""
            print(f"  [{h['distance']:.4f}]{score} {h['title']}")
        print()

    answer_text = generate_answer(question, hits)
    return answer_text, hits


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    ans, hits = answer(question)
    print("--- Answer ---")
    print(ans)
