# ESCO Occupation–Skill RAG

A small RAG (Retrieval-Augmented Generation) system that answers questions
about occupations and skills, built on ESCO taxonomy data (occupation ↔
skill) from TalentCLEF. Each occupation is merged into a single document
(essential + optional skills) before embedding and indexing.

## Structure

```
config.py           # paths, model, API key configuration
ingest.py           # step 1: build documents.jsonl from the 3 raw files
index_store.py      # step 2: embed + store in ChromaDB (local, persistent)
query.py            # step 3: retrieval + generation (Gemini)
generate_testset.py # auto-generate a test set from documents.jsonl (sanity check)
eval.py             # step 4: evaluate recall@k, MRR (supports --rerank)
diagnose_eval.py    # print detail for queries where the correct answer isn't top-1
rerank.py           # cross-encoder reranking module
```

## Setup

```bash
pip install -r requirements.txt
```

Place the 3 raw data files in a `data/` folder (or set the `TALENTCLEF_DATA_DIR`
environment variable to point to the folder containing them):
```
data/job2skill.tsv
data/jobid2terms.json
data/skillid2terms.json
```

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-key-here"      # macOS/Linux
$env:GEMINI_API_KEY = "your-key-here"      # PowerShell
```

(Optional) Use a fine-tuned embedding checkpoint instead of the base model:
```bash
export EMBEDDING_MODEL="./models/mpnet-mnrl-finetuned"
```

## Running the pipeline

```bash
# 1. Build documents from the raw data
python ingest.py

# 2. Embed + index into ChromaDB (run once, or whenever you change the embedding model)
python index_store.py

# 3. Ask a question
python query.py "What skills are needed for a technical director?"
```

## Evaluating retrieval

**Quick way (automatic, sanity check):** generate a test set from the
already-indexed data — each question targets one specific occupation, used
to verify baseline retrieval works correctly:
```bash
python generate_testset.py --n 30 --seed 42
python eval.py test_questions.jsonl
```

**More thorough way (hand-written, for a report/CV):** write more varied
questions yourself (comparing two jobs, ambiguous questions, questions with
no answer in the data...). Format:
```json
{"question": "What skills are needed for a technical director?", "relevant_ids": ["http://data.europa.eu/esco/occupation/00030d09-..."]}
```

Run:
```bash
python eval.py test_questions.jsonl
```

## Evaluation results (base vs fine-tuned embedding)

Test set: 30 auto-generated questions from `generate_testset.py` (seed=42),
each targeting exactly one occupation among the 3039 indexed.

| Model | Mean Recall@5 | MRR |
|---|---|---|
| Base (`all-mpnet-base-v2`) | 0.7333 | 0.6750 |
| Fine-tuned (MNRL on ESCO occupation–skill pairs) | 1.0000 | 0.8778 |
| **Fine-tuned + cross-encoder rerank** (`ms-marco-MiniLM-L-6-v2`, pool_k=20) | **1.0000** | **0.9733** |

Fine-tuning improved Recall@5 by +26.7 absolute points (+36% relative) and
MRR by +0.20 (+30% relative). All 8 questions completely missed by the base
model were narrow/technical occupation titles rarely used in natural
language (e.g. avionics technician, securities analyst, geotechnician, hand
lasting operator, boom operator, scraper operator, rental manager) — all of
them were correctly retrieved after fine-tuning, though some still had a
lower MRR of 0.33–0.5 (correct but not top-1, further improvable with
reranking).

**Known limitation:** Vietnamese-language questions perform noticeably
worse than English ones (cross-lingual gap) — the current embedding model
was only fine-tuned on English data. Possible fixes: use a multilingual
embedding model, or translate the question to English before retrieval.

## Analyzing low-MRR cases (correct but not top-1) + Rerank

See exactly which candidate is outranking the correct answer:
```bash
python diagnose_eval.py test_questions.jsonl
```

Try improving this with cross-encoder reranking (retrieve top 20, then
rerank down to top 5 before computing metrics):
```bash
python eval.py test_questions.jsonl --rerank --pool-k 20
```

Compare Mean Recall@5 / MRR between the two runs (with/without `--rerank`)
to decide whether reranking is worth adding to a production pipeline — the
trade-off is that each query now costs one extra cross-encoder pass over
`pool_k` candidates, slower than dense-only but still much faster than
calling the LLM.

**Actual result:** reranking improved MRR from 0.8778 to 0.9733 (+10.9%
relative) — 6 out of 7 cases that missed top-1 under dense-only retrieval
reached top-1 after reranking (avionics technician, securities analyst, tax
compliance officer, geotechnician, hand lasting operator, and one
satellite-engineer-type case).

**But one case got worse:** `import export manager` dropped from rank 2
(RR=0.50) to rank 5 (RR=0.20) after reranking. Reason: specialized variants
of the same occupation (`import export manager in fruit and vegetables`,
`...in beverages`, `...in flowers and plants`) have skill content almost
identical to the generic title — the cross-encoder, while generally more
accurate than dense embeddings, still can't distinguish a generic job title
from a narrowly specialized one when the text content is nearly identical.
This is a shared limitation of semantic search (both dense and
cross-encoder), not an implementation bug — a real fix would need a signal
beyond semantics (e.g. prioritizing exact/fuzzy title matching, or using
the ESCO hierarchy instead of relying purely on skill content).

## Future directions

- **Hybrid search**: add BM25 (rank_bm25) alongside dense retrieval, merge
  results (reciprocal rank fusion) — compare recall@k before/after.
- **Rerank**: use a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
  to rerank the top-20 candidates before passing the final top-5 into the prompt.
- **Query rewriting**: expand the question with synonyms before embedding,
  leveraging the `jobid2terms.json` / `skillid2terms.json` data itself.
- **Agentic step**: add a "filter by skill category" or "compare two job
  titles" tool — a stepping stone toward an AI agent.

## Related work

The embedding model used in this project was fine-tuned (MNRL on ESCO
occupation–skill pairs) as part of **TalentCLEF 2026 Task B** (Job-Skill
Matching with Skill Type Classification), CLEF 2026.

- Paper (working notes, CEUR-WS): [paper477](https://clef-staging.pages.dev/paper477.pdf)
- TalentCLEF overview: [Overview of the TalentCLEF 2026](https://www.emergentmind.com/papers/2606.31692)
