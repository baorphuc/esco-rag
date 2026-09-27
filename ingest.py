"""
Bước 1: Đọc job2skill.tsv + jobid2terms.json + skillid2terms.json
        -> gộp thành 1 document / occupation
        -> ghi ra documents.jsonl

Mỗi document có cấu trúc:
{
  "id": "<occupation_uri>",
  "title": "<label chính>",
  "text": "<đoạn text đầy đủ để embed>",
  "essential_skills": [...],
  "optional_skills": [...]
}
"""
import json
from collections import defaultdict

import ftfy

from config import (
    JOB2SKILL_TSV,
    JOBID2TERMS_JSON,
    SKILLID2TERMS_JSON,
    DOCUMENTS_JSONL,
    MAX_SYNONYMS_IN_DOC,
)


def load_terms(path: str) -> dict:
    """Load terms JSON, tự động fix mojibake (vd. â€™ -> ') trong dữ liệu ESCO gốc."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        uri: [ftfy.fix_text(t) for t in terms]
        for uri, terms in data.items()
    }


def format_label(terms: list[str]) -> str:
    """Label chính + tối đa MAX_SYNONYMS_IN_DOC synonym phụ trong ngoặc."""
    if not terms:
        return "unknown"
    main = terms[0]
    extra = terms[1 : 1 + MAX_SYNONYMS_IN_DOC]
    if extra:
        return f"{main} (also known as: {', '.join(extra)})"
    return main


def build_documents():
    print("Loading terms...")
    job_terms = load_terms(JOBID2TERMS_JSON)
    skill_terms = load_terms(SKILLID2TERMS_JSON)

    print("Reading job2skill.tsv...")
    job_to_essential = defaultdict(list)
    job_to_optional = defaultdict(list)

    with open(JOB2SKILL_TSV, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            job_uri, skill_uri, relation = parts
            skill_label = format_label(skill_terms.get(skill_uri, []))
            if relation.strip().lower() == "essential":
                job_to_essential[job_uri].append(skill_label)
            else:
                job_to_optional[job_uri].append(skill_label)

    print(f"Building documents for {len(job_terms)} occupations...")
    n_written = 0
    with open(DOCUMENTS_JSONL, "w", encoding="utf-8") as out:
        for job_uri, terms in job_terms.items():
            title = format_label(terms)
            essential = job_to_essential.get(job_uri, [])
            optional = job_to_optional.get(job_uri, [])

            text_parts = [f"Job: {title}"]
            if essential:
                text_parts.append("Essential skills:\n" + "\n".join(f"- {s}" for s in essential))
            if optional:
                text_parts.append("Optional skills:\n" + "\n".join(f"- {s}" for s in optional))
            text = "\n\n".join(text_parts)

            doc = {
                "id": job_uri,
                "title": terms[0] if terms else "unknown",
                "text": text,
                "essential_skills": essential,
                "optional_skills": optional,
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Done. Wrote {n_written} documents to {DOCUMENTS_JSONL}")


if __name__ == "__main__":
    build_documents()
