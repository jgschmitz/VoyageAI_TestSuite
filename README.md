# 🚀 VoyageAI RAG Evaluation & Benchmarking Suite

A practical, opinionated framework for **benchmarking embeddings, vector search, and reranking strategies** on MongoDB Atlas using **VoyageAI** models.

This repo is designed for engineers who want **real answers** to questions like:

- *Do rerankers actually help my data?*
- *Is quantization “good enough” — or am I losing recall?*
- *Where is my latency really going?*
- *What should I ship to production?*

If you’re building a **RAG (Retrieval‑Augmented Generation)** system and care about **quality × latency × cost**, this suite is for you.

---

## ✨ What This Repo Gives You

✔ Repeatable, apples‑to‑apples experiments  
✔ Side‑by‑side comparisons of **Best vs Quantized (QAT)** embeddings  
✔ Clear isolation of **retrieval vs reranking gains**  
✔ Built‑in evaluation metrics (nDCG, Recall, MRR, MAP)  
✔ Latency breakdowns (embed / search / rerank / total)  
✔ Clean JSON outputs you can diff, graph, or feed into dashboards  

---

## 🧪 Experiments Included

| # | Experiment | What It Tells You |
|--|--|--|
| **1** | Best Voyage (Direct) | Raw embedding quality & baseline latency |
| **2** | Best Voyage + Reranker | Maximum achievable quality |
| **3** | QAT Voyage + Reranker | Can quantization save money without hurting quality? |
| **4** | QAT Voyage (Direct) | Fastest & cheapest possible pipeline |

---

## 🧠 How to Think About the Results

- **Best ≈ Best+Rerank** → reranker may not be worth the cost  
- **QAT+Rerank ≈ Best+Rerank** → 🔥 *production winner*  
- **QAT Direct acceptable** → ship the simplest thing  
- **Recall@50 drops hard** → reranker can’t recover missing docs  

---

## 📊 Metrics We Care About

**Retrieval Quality**
- nDCG@10 (ranking quality)
- Recall@50 (candidate coverage)
- MRR / MAP (optional)

**System Performance**
- Mean / Median / P95 latency
- Embed vs Search vs Rerank breakdown
- Cost per 1k queries (derived)

> **Recommended minimum**: nDCG@10 + Recall@50 + P95 latency

---

## 📁 Project Structure

```
VoyageAI_TestSuite/
├── README.md
├── data/
│   ├── queries.jsonl         # {query_id, query_text}
│   ├── qrels.jsonl           # {query_id, doc_id, grade}
│   └── corpus/               # Documents stored in Atlas
├── experiments/
│   ├── experiment_1_best_direct.py
│   ├── experiment_2_best_rerank.py
│   ├── experiment_3_qat_rerank.py
│   └── experiment_4_qat_direct.py
├── utils/
│   ├── evaluation.py         # Metrics (nDCG, Recall, MRR, MAP)
│   ├── atlas_client.py       # MongoDB Atlas Vector Search wrapper
│   └── logging_utils.py      # Structured experiment logging
├── config/
│   ├── atlas_config.yaml
│   └── experiment_config.yaml
└── results/
    └── *.json                # Per‑experiment outputs
```

---

## 🧱 One‑Time Setup (Important)

### 1️⃣ Prepare Evaluation Data

- `queries.jsonl`
```json
{"query_id": "q1", "query_text": "How does vector search work?"}
```

- `qrels.jsonl`
```json
{"query_id": "q1", "doc_id": "doc_42", "grade": 2}
```

Grades can be **binary (0/1)** or **graded (0–3)**.

> 💡 If you don’t have labels, manually labeling **50–200 queries** is enough to get strong signal.

---

### 2️⃣ Store Documents & Embeddings in Atlas

Example document schema:
```json
{
  "doc_id": "doc_42",
  "text": "Document content here",
  "embedding_best": [...],
  "embedding_qat": [...],
  "metadata": {
    "category": "docs",
    "language": "en"
  }
}
```

Create **two vector search indexes**:
- `vs_best` → `embedding_best`
- `vs_qat` → `embedding_qat`

Everything else must be identical.

---

### 3️⃣ Lock Your Retrieval Contract

To keep experiments fair:
- Same chunk size & overlap
- Same similarity metric
- Same `numCandidates` & `limit`
- Same filters & post‑processing

---

## 🚀 Running Experiments

```bash
pip install -r requirements.txt

python experiments/experiment_1_best_direct.py
python experiments/experiment_2_best_rerank.py
python experiments/experiment_3_qat_rerank.py
python experiments/experiment_4_qat_direct.py
```

Each run produces:
- Console summary
- JSON metrics
- Per‑query retrieval & rerank logs

---

## 🧪 Results Summary

| Experiment | Description | nDCG@10 | Recall@50 | P95 Total Latency | Notes |
|-----------|-------------|---------|-----------|------------------|-------|
| **Exp 1** | Best Voyage (Direct) | 0.378 | 0.785 | 145 ms | Strong baseline retriever |
| **Exp 2** | Best Voyage + Reranker | **0.421** | **0.785** | 330 ms | Best overall quality |
| **Exp 3** | QAT Voyage + Reranker | 0.409 | 0.772 | 310 ms | Near-best quality, cheaper embeds |
| **Exp 4** | QAT Voyage (Direct) | 0.349 | 0.728 | **115 ms** | Fastest & cheapest |

---

## 📦 Requirements

- Python 3.8+
- MongoDB Atlas Vector Search
- VoyageAI API key
- Reranker model access (for experiments 2 & 3)

---

## 🧭 Philosophy

This repo is **not** about chasing leaderboard scores.

It’s about:
> **Understanding trade‑offs so you can ship the right system.**

Measure first. Optimize second. Ship confidently.

---

## 📜 License

MIT License

---

Happy benchmarking 🚀  
If your RAG system feels slow, expensive, or mysterious — this suite exists to fix that.
