"""
Tạo test set cho eval.py bằng cách sample ngẫu nhiên N occupation thật
từ documents.jsonl, rồi sinh câu hỏi theo template.

Vì mỗi câu hỏi nhắm thẳng vào 1 occupation cụ thể, relevant_ids chỉ có
đúng 1 phần tử là chính occupation đó -> đây là easy case, dùng để
kiểm tra baseline retrieval hoạt động đúng cơ bản (sanity check), không
thay thế cho một test set được viết tay có nhiều edge case hơn.

Chạy: python generate_testset.py --n 30 --seed 42
"""
import argparse
import json
import random

from config import DOCUMENTS_JSONL

# Nhiều template khác nhau để câu hỏi không bị lặp 1 kiểu,
# và để kiểm tra retrieval có robust với cách hỏi khác nhau không.
TEMPLATES = [
    "What skills are needed for a {title}?",
    "What are the essential skills for a {title}?",
    "List the skills required to become a {title}.",
    "What competencies does a {title} need?",
    "What does a {title} need to know how to do?",
]


def load_documents(path: str):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Số câu hỏi cần sinh")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="test_questions.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    docs = load_documents(DOCUMENTS_JSONL)

    # chỉ lấy occupation có ít nhất 1 essential skill, tránh document rỗng/nghèo thông tin
    docs = [d for d in docs if d.get("essential_skills")]
    sample = random.sample(docs, min(args.n, len(docs)))

    with open(args.out, "w", encoding="utf-8") as out:
        for doc in sample:
            template = random.choice(TEMPLATES)
            question = template.format(title=doc["title"])
            item = {
                "question": question,
                "relevant_ids": [doc["id"]],
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} questions to {args.out}")


if __name__ == "__main__":
    main()
