"""
Cấu hình chung cho pipeline RAG.
Sửa các đường dẫn / model tại đây, không cần đụng vào code chính.
"""
import os

# ---- Dữ liệu nguồn (đổi theo máy bạn) ----
DATA_DIR = os.environ.get("TALENTCLEF_DATA_DIR", "./data")
JOB2SKILL_TSV = os.path.join(DATA_DIR, "job2skill.tsv")
JOBID2TERMS_JSON = os.path.join(DATA_DIR, "jobid2terms.json")
SKILLID2TERMS_JSON = os.path.join(DATA_DIR, "skillid2terms.json")

# ---- Output ----
DOCUMENTS_JSONL = "./documents.jsonl"
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "esco_occupations")

# ---- Embedding model ----
# Đổi thành đường dẫn checkpoint fine-tune local nếu có, vd:
# EMBEDDING_MODEL = "./models/mpnet-mnrl-finetuned"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# ---- Document building ----
MAX_SYNONYMS_IN_DOC = 3   # số synonym phụ hiển thị trong document (ngoài label chính)
TOP_K_RETRIEVAL = 5
TOP_K_POOL = 20           # số candidate retrieve trước khi rerank xuống top k

# ---- Gemini (generation) ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"
