"""
Bước 2: Đọc documents.jsonl -> embed -> lưu vào ChromaDB (persistent, local).

Chạy: python index_store.py
"""
import json

import chromadb
from sentence_transformers import SentenceTransformer

from config import DOCUMENTS_JSONL, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def load_documents():
    docs = []
    with open(DOCUMENTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading documents...")
    docs = load_documents()
    print(f"{len(docs)} documents loaded.")

    print("Encoding (may take a while on CPU)...")
    texts = [d["text"] for d in docs]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print("Writing to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # xóa collection cũ nếu re-index
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma giới hạn batch add, chia nhỏ cho an toàn
    BATCH = 512
    for i in range(0, len(docs), BATCH):
        batch_docs = docs[i : i + BATCH]
        collection.add(
            ids=[d["id"] for d in batch_docs],
            embeddings=embeddings[i : i + BATCH],
            documents=[d["text"] for d in batch_docs],
            metadatas=[
                {
                    "title": d["title"],
                    "essential_skills": json.dumps(d["essential_skills"], ensure_ascii=False),
                    "optional_skills": json.dumps(d["optional_skills"], ensure_ascii=False),
                }
                for d in batch_docs
            ],
        )
        print(f"  indexed {min(i + BATCH, len(docs))}/{len(docs)}")

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count()} items in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
