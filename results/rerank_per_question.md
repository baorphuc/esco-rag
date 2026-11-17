# Rerank analysis (fine-tuned embedding, per question)
- Test set: test_questions.jsonl (N=30)
- Pool-k: 20, final k: 5
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Run at: 2026-11-17T17:19:07.056678

## Summary
- Recall@5 before rerank (in pool): 1.0000
- Recall@5 after rerank:           1.0000
- MRR before rerank:                 0.8778
- MRR after rerank:                  0.9733
- Improved (rank moved up):          5
- Worse (rank moved down):           1
- Newly found (missed before, found after): 0
- Lost (found before, missed after): 0

## Per-question detail
| # | Question | Relevant | Rank before | Rank after | RR before | RR after | Delta | Status |
|---|----------|----------|-------------|------------|-----------|----------|-------|--------|
| 1 | What does a avionics technician need to know how to do? | avionics technician | 2 | 1 | 0.50 | 1.00 | +0.50 | improved |
| 2 | List the skills required to become a satellite engineer. | satellite engineer | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 3 | What skills are needed for a securities analyst? | securities analyst | 3 | 1 | 0.33 | 1.00 | +0.67 | improved |
| 4 | What are the essential skills for a intellectual property consultant? | intellectual property consultant | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 5 | What competencies does a fiscal affairs policy officer need? | fiscal affairs policy officer | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 6 | List the skills required to become a tax compliance officer. | tax compliance officer | 3 | 1 | 0.33 | 1.00 | +0.67 | improved |
| 7 | List the skills required to become a commercial director. | commercial director | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 8 | What are the essential skills for a microelectronics maintenance technician? | microelectronics maintenance technician | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 9 | What are the essential skills for a geotechnician? | geotechnician | 3 | 1 | 0.33 | 1.00 | +0.67 | improved |
| 10 | List the skills required to become a import export manager. | import export manager | 2 | 5 | 0.50 | 0.20 | -0.30 | worse |
| 11 | What skills are needed for a electrical equipment inspector? | electrical equipment inspector | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 12 | What skills are needed for a checkout supervisor? | checkout supervisor | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 13 | What competencies does a architect need? | architect | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 14 | What skills are needed for a hand lasting operator? | hand lasting operator | 3 | 1 | 0.33 | 1.00 | +0.67 | improved |
| 15 | List the skills required to become a economics lecturer. | economics lecturer | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 16 | List the skills required to become a power lines supervisor. | power lines supervisor | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 17 | What does a wood and construction materials distribution manager need to know how to do? | wood and construction materials distribution manager | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 18 | List the skills required to become a boom operator. | boom operator | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 19 | What skills are needed for a passport officer? | passport officer | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 20 | What competencies does a scraper operator need? | scraper operator | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 21 | What does a dependability engineer need to know how to do? | dependability engineer | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 22 | What skills are needed for a furniture carpets and lighting equipment distribution manager? | furniture carpets and lighting equipment distribution manager | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 23 | What competencies does a decorative painter need? | decorative painter | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 24 | What skills are needed for a bodyguard? | bodyguard | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 25 | What does a rental manager need to know how to do? | rental manager | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 26 | List the skills required to become a aeronautical information specialist. | aeronautical information specialist | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 27 | What does a tramways technician need to know how to do? | tramways technician | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 28 | List the skills required to become a plastic rolling machine operator. | plastic rolling machine operator | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 29 | What does a lottery manager need to know how to do? | lottery manager | 1 | 1 | 1.00 | 1.00 | +0.00 | same |
| 30 | What are the essential skills for a language school teacher? | language school teacher | 1 | 1 | 1.00 | 1.00 | +0.00 | same |

## Cases that got WORSE after rerank
- "List the skills required to become a import export manager." (relevant: import export manager / http://data.europa.eu/esco/occupation/b87a7a27-ee74-4e5e-b274-84a3cab27bf4): rank 2 -> 5, RR 0.50 -> 0.20

## Cases that got BETTER after rerank
- "What does a avionics technician need to know how to do?" (relevant: avionics technician / http://data.europa.eu/esco/occupation/d9ab0f3d-078d-42a7-a502-3e739e0aeff6): rank 2 -> 1, RR 0.50 -> 1.00
- "What skills are needed for a securities analyst?" (relevant: securities analyst / http://data.europa.eu/esco/occupation/076d23f3-a689-4f1a-8b55-ae870c9aaa85): rank 3 -> 1, RR 0.33 -> 1.00
- "List the skills required to become a tax compliance officer." (relevant: tax compliance officer / http://data.europa.eu/esco/occupation/4a39fddb-17e7-4431-be03-613edab8d1c8): rank 3 -> 1, RR 0.33 -> 1.00
- "What are the essential skills for a geotechnician?" (relevant: geotechnician / http://data.europa.eu/esco/occupation/e9665bdf-2989-4dc5-9aa9-7b791e489779): rank 3 -> 1, RR 0.33 -> 1.00
- "What skills are needed for a hand lasting operator?" (relevant: hand lasting operator / http://data.europa.eu/esco/occupation/0a5001af-ce4b-4787-92e8-62af70089e53): rank 3 -> 1, RR 0.33 -> 1.00
