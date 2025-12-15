# VoyageAI Testing Suite

A comprehensive framework for systematically evaluating different embedding models, retrieval strategies, and reranking approaches to optimize your RAG (Retrieval-Augmented Generation) pipeline.

## Overview

This testing suite enables you to run controlled A/B experiments comparing:
- **Voyage Best** (full-precision) vs **Voyage QAT** (quantized) embedding models
- **Direct retrieval** vs **Retrieval + Reranking** strategies
- **Performance trade-offs** between quality, latency, and cost

## Shared Setup (Do This Once for All Experiments)

### A) Define What You're Measuring

Pick **1 primary objective** and stick to it across all variants:

**Retrieval Quality Options:**
- **Recall@K, nDCG@K, MRR** (requires relevance labels)
- **RAG answer quality**: exact match / rubric grading (slower, more subjective)
- **System metrics**: p50/p95 latency, $ cost per query, index size, memory

**Recommended Minimal Set:**
- **nDCG@10** (quality)
- **Recall@50** (did we fetch the right stuff at all?)
- **p95 latency end-to-end** (user experience)
- **cost per 1k queries** (real-world viability)

### B) Build an Evaluation Dataset

Create the following files:

1. **`queries.jsonl`**: `{query_id, query_text}`
2. **`qrels.jsonl`**: `{query_id, doc_id, grade}` (grade can be 0/1 or 0–3)
3. **`corpus`**: documents already in Atlas with a stable `doc_id`

**If you don't have labels**, you can bootstrap:
- Use historical clicks / "was this helpful" signals, or
- Manually label ~50–200 queries (small but very useful)

### C) In Atlas: Store Documents + Embeddings

**Collection Example Fields:**
```javascript
{
  doc_id: "unique_identifier",
  text: "document_content",
  embedding_best: [vector_data],       // Full-precision Voyage model
  embedding_qat: [vector_data],        // Quantized Voyage model (optional)
  metadata: {
    tenant: "...",
    category: "...",
    language: "...",
    // ... other filter fields
  }
}
```

**Create Two Vector Search Indexes** (recommended for clean A/B):
- **Index `vs_best`** pointing to `embedding_best`
- **Index `vs_qat`** pointing to `embedding_qat`

Keep everything else identical (same tokenization, same chunking, same doc set).

### D) Fix Your "Retrieval Contract"

Lock these parameters for fairness:
- **Chunking strategy** (size/overlap)
- **Similarity metric** (cosine vs dot)
- **numCandidates and limit** (e.g., numCandidates=200, limit=50)
- **Filters** (if any)
- **Post-processing** (dedupe, max per source, etc.)

### E) Logging (Very Important)

For every query + variant, log:
- **Variant name**
- **Retrieved doc_ids with scores**
- **Reranked doc_ids with rerank scores** (if used)
- **Latency breakdown**: embed, Atlas search, rerank, total
- **Model/version identifiers**

---

## Experiment 1 — Best Voyage Model Direct Output (No Reranker)

### Goal
Baseline retrieval quality + latency using your best embedding model.

### Steps
1. **Embed the query** using the Best Voyage model → `q_vec_best`
2. **Atlas Vector Search** on `vs_best`:
   - Set `limit = K` (e.g., 10 or 20 for final results)
   - Set `numCandidates` higher (e.g., 200) for better recall
3. **Return the top-K results** directly
4. **Evaluate metrics** (nDCG@10, Recall@50, p95 latency, etc.)

### Notes
- This isolates "raw retriever quality"
- If this is weak, reranking won't fully save it

---

## Experiment 2 — Best Voyage + Rescorer Output (Reranker)

### Goal
Measure how much reranking improves quality, and what it costs in latency.

### Steps
1. **Embed query** with Best Voyage → `q_vec_best`
2. **Retrieve top-N candidates** from Atlas (bigger than K):
   - e.g., N = 50 or 100
3. **Rerank those N** using the Rescorer:
   - Input pairs: `(query_text, candidate_text)`
   - Output: rerank scores
4. **Return the top-K** after rerank (K=10/20)
5. **Evaluate the same metrics**

### Notes
- Choose N so reranker has enough room to fix ordering (50 is common)
- **Track:**
  - Quality delta (e.g., nDCG@10 improves?)
  - Latency delta (rerank is usually the expensive step)

---

## Experiment 3 — QAT (Quantized) Voyage + Rescorer Output

### Goal
See if quantizing the embedding model reduces cost/latency while maintaining quality (with reranker safety net).

### Steps
1. **Embed query** with the quantized/QAT Voyage model → `q_vec_qat`
2. **Atlas Vector Search** on `vs_qat` retrieving top-N candidates:
   - Keep N, numCandidates identical to Experiment 2
3. **Rerank the same way** with the Rescorer
4. **Return top-K** after rerank
5. **Evaluate metrics** + latency + cost

### Notes
- Reranker often masks small embedding degradation
- If QAT retriever drops Recall@50 a lot, reranker can't recover missing docs

---

## Experiment 4 — Simple QAT (Quantized) Voyage Direct Output (No Reranker)

### Goal
Best-case "cheap" pipeline: quantized embeddings only.

### Steps
1. **Embed query** with QAT model → `q_vec_qat`
2. **Atlas Vector Search** on `vs_qat` with `limit=K`
3. **Return top-K** directly
4. **Evaluate metrics**

### Notes
- This is usually fastest/cheapest
- Often the biggest quality tradeoff—good to know if it's "good enough"

---

## How to Keep the Comparison Fair

- **Same corpus**, same chunking, same filters, same K/N/numCandidates
- **Same evaluation set** (queries + qrels)
- **Warm up the system** (run a few hundred queries first) before collecting latency stats
- **Run enough queries** (ideally 200+ labeled queries) to avoid noisy conclusions

## Suggested "Default" Parameter Choices (Good Starting Point)

- **K** (final shown): 10
- **N** (candidates to rerank): 50
- **numCandidates**: 200
- **Metrics**: nDCG@10, Recall@50, p95 latency
- **Report**: mean + median + p95

---

## Quick Interpretation Guide

- **If Best direct ≈ Best+rerank**: reranker might not be worth the cost
- **If QAT+rerank ≈ Best+rerank** on nDCG@10 but cheaper/faster → **strong win**
- **If QAT direct is acceptable** and you need speed → **simplest deployment**
- **If QAT direct loses Recall@50 heavily**, don't ship it without rerank

---

## Project Structure

```
VoyageAI_TestSuite/
├── README.md                 # This file
├── data/
│   ├── queries.jsonl         # Test queries
│   ├── qrels.jsonl           # Relevance judgments
│   └── corpus/               # Document corpus
├── experiments/
│   ├── experiment_1_best_direct.py
│   ├── experiment_2_best_rerank.py
│   ├── experiment_3_qat_rerank.py
│   └── experiment_4_qat_direct.py
├── utils/
│   ├── evaluation.py         # Metrics calculation
│   ├── atlas_client.py       # Atlas Vector Search client
│   └── logging_utils.py      # Experiment logging
├── config/
│   ├── atlas_config.yaml     # Atlas connection settings
│   └── experiment_config.yaml # Experiment parameters
└── results/
    ├── experiment_1_results.json
    ├── experiment_2_results.json
    ├── experiment_3_results.json
    └── experiment_4_results.json
```

## Getting Started

1. **Set up your Atlas database** following section C above
2. **Prepare your evaluation dataset** following section B above
3. **Configure your experiments** in `config/experiment_config.yaml`
4. **Run experiments sequentially** using the provided scripts
5. **Compare results** using the evaluation metrics

## Requirements

- Python 3.8+
- Atlas Vector Search access
- Voyage AI API access
- Rescorer model access (for reranking experiments)

## Installation

```bash
git clone https://github.com/jgschmitz/VoyageAI_TestSuite.git
cd VoyageAI_TestSuite
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[MIT License](LICENSE)

---

**Happy experimenting!** 🚀

Remember: The goal is not just to find the "best" configuration, but to understand the trade-offs between quality, cost, and latency for your specific use case.
