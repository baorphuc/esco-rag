"""
Chẩn đoán các câu hỏi mà câu trả lời đúng có trong top-k nhưng không phải
top-1 (RR < 1.0) -- in đầy đủ ranking để xem cái gì đang xếp trên nó.

Chạy: python diagnose_eval.py test_questions.jsonl
"""
import json
import sys

from query import retrieve
from config import TOP_K_RETRIEVAL


def load_testset(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main(testset_path: str, k: int = TOP_K_RETRIEVAL):
    items = load_testset(testset_path)

    for item in items:
        hits = retrieve(item["question"], k=k)
        retrieved_ids = [h["id"] for h in hits]
        relevant = set(item["relevant_ids"])

        # tìm rank của đáp án đúng (1-indexed), None nếu không có trong top-k
        correct_rank = None
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant:
                correct_rank = rank
                break

        # chỉ in các case "gần đúng nhưng chưa top-1": có trong top-k nhưng rank > 1
        if correct_rank is not None and correct_rank > 1:
            print(f"\n=== Q: {item['question']} ===")
            print(f"Đáp án đúng đứng hạng {correct_rank}/{k}")
            for i, h in enumerate(hits, 1):
                marker = " <-- ĐÁP ÁN ĐÚNG" if h["id"] in relevant else ""
                print(f"  {i}. [{h['distance']:.4f}] {h['title']}{marker}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_eval.py test_questions.jsonl")
        sys.exit(1)
    main(sys.argv[1])
