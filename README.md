# ESCO Occupation–Skill RAG

RAG nhỏ trả lời câu hỏi về nghề nghiệp và kỹ năng, dựa trên dữ liệu ESCO
(occupation ↔ skill) từ TalentCLEF. Mỗi occupation được gộp thành 1 document
(essential + optional skills) trước khi embed và index.

## Cấu trúc

```
config.py           # cấu hình đường dẫn, model, API key
ingest.py           # bước 1: build documents.jsonl từ 3 file gốc
index_store.py      # bước 2: embed + lưu vào ChromaDB (local, persistent)
query.py            # bước 3: retrieval + generation (Gemini)
generate_testset.py # sinh test set tự động từ documents.jsonl (sanity check)
eval.py             # bước 4: đánh giá recall@k, MRR (có flag --rerank)
diagnose_eval.py    # in chi tiết các câu retrieval đúng nhưng chưa top-1
rerank.py           # module rerank bằng cross-encoder
```

## Setup

```bash
pip install -r requirements.txt
```

Đặt 3 file dữ liệu gốc vào thư mục `data/` (hoặc set biến môi trường
`TALENTCLEF_DATA_DIR` trỏ tới thư mục chứa chúng):
```
data/job2skill.tsv
data/jobid2terms.json
data/skillid2terms.json
```

Set API key cho Gemini:
```bash
export GEMINI_API_KEY="your-key-here"      # macOS/Linux
$env:GEMINI_API_KEY = "your-key-here"      # PowerShell
```

(Tùy chọn) Dùng embedding checkpoint đã fine-tune thay vì model gốc:
```bash
export EMBEDDING_MODEL="./models/mpnet-mnrl-finetuned"
```

## Chạy pipeline

```bash
# 1. Build documents từ dữ liệu gốc
python ingest.py

# 2. Embed + index vào ChromaDB (chạy 1 lần, hoặc mỗi khi đổi embedding model)
python index_store.py

# 3. Hỏi đáp
python query.py "Kỹ năng nào cần thiết cho technical director?"
```

## Đánh giá retrieval

**Cách nhanh (tự động, sanity check):** sinh test set từ chính dữ liệu đã index —
mỗi câu hỏi nhắm 1 occupation cụ thể, dùng để kiểm tra baseline retrieval hoạt
động đúng cơ bản:
```bash
python generate_testset.py --n 30 --seed 42
python eval.py test_questions.jsonl
```

**Cách kỹ hơn (viết tay, cho báo cáo/CV):** tự viết câu hỏi đa dạng hơn
(so sánh 2 job, câu hỏi mơ hồ, câu hỏi không có trong data...). Format:
```json
{"question": "Kỹ năng nào cần cho technical director?", "relevant_ids": ["http://data.europa.eu/esco/occupation/00030d09-..."]}
```

Chạy:
```bash
python eval.py test_questions.jsonl
```

## Kết quả đánh giá (base vs fine-tuned embedding)

Test set: 30 câu hỏi tự sinh từ `generate_testset.py` (seed=42), mỗi câu
nhắm đúng 1 occupation trong 3039 occupation đã index.

| Model | Mean Recall@5 | MRR |
|---|---|---|
| Base (`all-mpnet-base-v2`) | 0.7333 | 0.6750 |
| Fine-tuned (MNRL trên ESCO occupation–skill pairs) | 1.0000 | 0.8778 |
| **Fine-tuned + cross-encoder rerank** (`ms-marco-MiniLM-L-6-v2`, pool_k=20) | **1.0000** | **0.9733** |

Fine-tuning cải thiện Recall@5 tuyệt đối +26.7 điểm (+36% relative), MRR
+0.20 (+30% relative). Toàn bộ 8 câu bị miss hoàn toàn ở model gốc — đều là
occupation có tên hẹp/kỹ thuật, ít xuất hiện trong ngôn ngữ tự nhiên (vd.
avionics technician, securities analyst, geotechnician, hand lasting
operator, boom operator, scraper operator, rental manager) — đều được
retrieve đúng sau fine-tune, dù một số câu MRR còn ở mức 0.33-0.5 (đúng
nhưng chưa phải top-1, có thể cải thiện thêm bằng rerank).

**Hạn chế đã ghi nhận:** câu hỏi tiếng Việt cho kết quả retrieval kém hơn
hẳn tiếng Anh (cross-lingual gap) — model embedding hiện tại chỉ fine-tune
trên dữ liệu tiếng Anh. Hướng khắc phục: dùng multilingual embedding model,
hoặc dịch câu hỏi sang tiếng Anh trước khi retrieve.

## Phân tích các case MRR thấp (đúng nhưng chưa top-1) + Rerank

Xem chi tiết candidate nào đang xếp trên đáp án đúng:
```bash
python diagnose_eval.py test_questions.jsonl
```

Thử cải thiện bằng cross-encoder rerank (retrieve top 20 rồi rerank
xuống top 5 trước khi tính metric):
```bash
python eval.py test_questions.jsonl --rerank --pool-k 20
```

So sánh Mean Recall@5 / MRR giữa 2 lần chạy (có/không `--rerank`) để biết
rerank có đáng thêm vào pipeline production hay không — đánh đổi là mỗi
query giờ tốn thêm 1 lượt gọi cross-encoder cho `pool_k` candidate,
chậm hơn dense-only nhưng vẫn nhanh hơn nhiều so với gọi LLM.

**Kết quả thực tế:** rerank cải thiện MRR từ 0.8778 lên 0.9733 (+10.9%
relative) — 6/7 case bị miss top-1 ở dense-only đều lên đúng top-1 sau
rerank (avionics technician, securities analyst, tax compliance officer,
geotechnician, hand lasting operator, satellite engineer-type cases).

**Nhưng có 1 case tệ hơn:** `import export manager` từ hạng 2 (RR=0.50)
tụt xuống hạng 5 (RR=0.20) sau rerank. Nguyên nhân: các biến thể chuyên
môn hóa của cùng occupation (`import export manager in fruit and
vegetables`, `...in beverages`, `...in flowers and plants`) có nội dung
skill gần như trùng lặp với bản gốc — cross-encoder, dù chính xác hơn
dense embedding nói chung, vẫn không phân biệt được job tổng quát với
job chuyên môn hóa hẹp khi text content quá giống nhau. Đây là giới hạn
chung của semantic search (cả dense lẫn cross-encoder) chứ không phải lỗi
implementation — hướng khắc phục thực sự cần thêm tín hiệu ngoài semantic
(vd. ưu tiên exact/fuzzy match tên occupation, hoặc phân biệt theo cấu
trúc phân cấp ESCO thay vì chỉ dựa vào nội dung skill).

Kết quả in ra recall@k và MRR — dùng để so sánh các phương án khác nhau
(embedding model gốc vs fine-tuned, k khác nhau, có/không hybrid search, v.v.)

## Hướng mở rộng

- **Hybrid search**: thêm BM25 (rank_bm25) song song với dense retrieval,
  merge kết quả (reciprocal rank fusion) — so sánh recall@k trước/sau.
- **Rerank**: dùng cross-encoder (vd. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
  để rerank top-20 candidates trước khi đưa 5 candidates cuối vào prompt.
- **Query rewriting**: mở rộng câu hỏi bằng synonym trước khi embed, tận
  dụng chính dữ liệu `jobid2terms.json` / `skillid2terms.json`.
- **Agentic step**: thêm tool "filter theo skill category" hoặc "so sánh
  2 job title" — bước đệm để nâng cấp thành AI agent.
