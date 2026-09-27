"""
Bước 4 (điểm nhấn CV): đánh giá retrieval bằng recall@k và MRR.

Format file test set (JSONL), mỗi dòng:
{"question": "...", "relevant_ids": ["<occupation_uri>", ...]}

Bạn tự tạo file này bằng cách chọn ngẫu nhiên N occupation, viết câu hỏi
tương ứng (vd. "Kỹ năng nào cần cho <title>?") và ghi occupation_uri
làm relevant_id đúng. Hoặc dùng generate_testset.py để sinh tự động.

Chạy (chỉ dense retrieval):
    python eval.py test_questions.jsonl

Chạy có rerank (retrieve top --pool-k rồi rerank bằng cross-encoder,
lấy top k cuối để tính metric -- chậm hơn vì phải gọi cross-encoder
cho từng candidate, nhưng thường tăng MRR đáng kể):
    python eval.py test_questions.jsonl --rerank --pool-k 20
"""
import argparse
import json

from query import retrieve
from config import TOP_K_RETRIEVAL


def load_testset(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hit = len(set(retrieved_ids) & relevant_ids)
    return hit / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(testset_path: str, k: int = TOP_K_RETRIEVAL, use_rerank: bool = False, pool_k: int = 20):
    items = load_testset(testset_path)
    recalls, rrs = [], []

    if use_rerank:
        from rerank import rerank as rerank_fn

    for item in items:
        # nếu rerank: lấy pool lớn hơn (pool_k) trước, rồi rerank xuống còn k
        retrieve_k = pool_k if use_rerank else k
        hits = retrieve(item["question"], k=retrieve_k)

        if use_rerank:
            hits = rerank_fn(item["question"], hits, top_k=k)

        retrieved_ids = [h["id"] for h in hits]
        relevant = set(item["relevant_ids"])

        r = recall_at_k(retrieved_ids, relevant)
        rr = reciprocal_rank(retrieved_ids, relevant)
        recalls.append(r)
        rrs.append(rr)

        print(f"Q: {item['question'][:60]:<60} recall@{k}={r:.2f}  RR={rr:.2f}")

    tag = "with rerank" if use_rerank else "dense only"
    print(f"\n=== Summary ({tag}) ===")
    print(f"Mean Recall@{k}: {sum(recalls) / len(recalls):.4f}")
    print(f"MRR:            {sum(rrs) / len(rrs):.4f}")
    print(f"N questions:    {len(items)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testset_path")
    parser.add_argument("--rerank", action="store_true", help="Bật cross-encoder rerank")
    parser.add_argument("--pool-k", type=int, default=20, help="Số candidate retrieve trước khi rerank")
    parser.add_argument("--k", type=int, default=TOP_K_RETRIEVAL, help="Số kết quả cuối để tính metric")
    args = parser.parse_args()

    evaluate(args.testset_path, k=args.k, use_rerank=args.rerank, pool_k=args.pool_k)

