"""
Bước 4: Phân tích rerank theo từng câu hỏi (cấu hình fine-tune).
In rank của đáp án đúng trước và sau cross-encoder rerank (pool 20 -> top 5),
liệt kê câu nào được cải thiện và câu nào bị tệ đi.
Luu ket qua vao results/rerank_per_question.md.

Chay: python rerank_per_question.py
  (can dat CHROMA_DIR=./chroma_db_ft, EMBEDDING_MODEL=<ft checkpoint>)
"""
import json
import os
import sys
from datetime import datetime

from query import retrieve
from rerank import rerank
from config import TOP_K_RETRIEVAL, TOP_K_POOL

POOL_K = TOP_K_POOL   # 20
K = TOP_K_RETRIEVAL   # 5


def load_testset(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def rank_of(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return rank
    return None


def main(testset_path: str = "test_questions.jsonl"):
    items = load_testset(testset_path)
    rows = []
    for item in items:
        relevant = set(item["relevant_ids"])
        hits = retrieve(item["question"], k=POOL_K)
        retrieved_ids = [h["id"] for h in hits]

        rank_before = rank_of(retrieved_ids, relevant)
        rr_before = 1.0 / rank_before if rank_before else 0.0

        reranked = rerank(item["question"], hits, top_k=K)
        reranked_ids = [h["id"] for h in reranked]
        rank_after = rank_of(reranked_ids, relevant)
        rr_after = 1.0 / rank_after if rank_after else 0.0

        rows.append({
            "question": item["question"],
            "relevant_id": sorted(relevant)[0],
            "relevant_title": None,
            "rank_before": rank_before,
            "rank_after": rank_after,
            "rr_before": rr_before,
            "rr_after": rr_after,
            "delta_rr": round(rr_after - rr_before, 4),
            "improved": (rank_after is not None and rank_before is not None and rank_after < rank_before),
            "worse": (rank_after is not None and rank_before is not None and rank_after > rank_before),
            "missed_before": rank_before is None,
            "missed_after": rank_after is None,
        })
        for h in hits:
            if h["id"] in relevant:
                rows[-1]["relevant_title"] = h["title"]
                break

    n = len(rows)
    recall_before = sum(1 for r in rows if r["rank_before"] is not None and r["rank_before"] <= K) / n
    recall_after = sum(1 for r in rows if r["rank_after"] is not None) / n
    mrr_before = sum(r["rr_before"] for r in rows) / n
    mrr_after = sum(r["rr_after"] for r in rows) / n

    improved = [r for r in rows if r["improved"]]
    worse = [r for r in rows if r["worse"]]
    newly_found = [r for r in rows if r["missed_before"] and not r["missed_after"]]
    lost = [r for r in rows if not r["missed_before"] and r["missed_after"]]

    lines = []
    lines.append("# Rerank analysis (fine-tuned embedding, per question)")
    lines.append(f"- Test set: {testset_path} (N={n})")
    lines.append(f"- Pool-k: {POOL_K}, final k: {K}")
    lines.append(f"- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
    lines.append(f"- Run at: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Recall@{K} before rerank (in pool): {recall_before:.4f}")
    lines.append(f"- Recall@{K} after rerank:           {recall_after:.4f}")
    lines.append(f"- MRR before rerank:                 {mrr_before:.4f}")
    lines.append(f"- MRR after rerank:                  {mrr_after:.4f}")
    lines.append(f"- Improved (rank moved up):          {len(improved)}")
    lines.append(f"- Worse (rank moved down):           {len(worse)}")
    lines.append(f"- Newly found (missed before, found after): {len(newly_found)}")
    lines.append(f"- Lost (found before, missed after): {len(lost)}")
    lines.append("")

    def fmt_rank(r):
        return str(r) if r is not None else "miss"

    lines.append("## Per-question detail")
    lines.append("| # | Question | Relevant | Rank before | Rank after | RR before | RR after | Delta | Status |")
    lines.append("|---|----------|----------|-------------|------------|-----------|----------|-------|--------|")
    for i, r in enumerate(rows, 1):
        q = r["question"].replace("|", "\\|")
        title = (r["relevant_title"] or "").replace("|", "\\|")
        if r["improved"]:
            status = "improved"
        elif r["worse"]:
            status = "worse"
        elif r["missed_before"] and not r["missed_after"]:
            status = "newly found"
        elif not r["missed_before"] and r["missed_after"]:
            status = "lost"
        else:
            status = "same"
        lines.append(f"| {i} | {q} | {title} | {fmt_rank(r['rank_before'])} | {fmt_rank(r['rank_after'])} | {r['rr_before']:.2f} | {r['rr_after']:.2f} | {r['delta_rr']:+.2f} | {status} |")

    lines.append("")
    lines.append("## Cases that got WORSE after rerank")
    for r in worse:
        lines.append(f"- \"{r['question']}\" (relevant: {r['relevant_title']} / {r['relevant_id']}): "
                     f"rank {fmt_rank(r['rank_before'])} -> {fmt_rank(r['rank_after'])}, "
                     f"RR {r['rr_before']:.2f} -> {r['rr_after']:.2f}")
    if not worse:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Cases that got BETTER after rerank")
    for r in improved:
        lines.append(f"- \"{r['question']}\" (relevant: {r['relevant_title']} / {r['relevant_id']}): "
                     f"rank {fmt_rank(r['rank_before'])} -> {fmt_rank(r['rank_after'])}, "
                     f"RR {r['rr_before']:.2f} -> {r['rr_after']:.2f}")
    if not improved:
        lines.append("- (none)")

    os.makedirs("results", exist_ok=True)
    out_path = "results/rerank_per_question.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print(f"Recall@{K} before={recall_before:.4f} after={recall_after:.4f} | "
          f"MRR before={mrr_before:.4f} after={mrr_after:.4f} | "
          f"improved={len(improved)} worse={len(worse)} newly_found={len(newly_found)} lost={len(lost)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_questions.jsonl")
