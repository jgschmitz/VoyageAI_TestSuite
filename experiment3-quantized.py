#!/usr/bin/env python3
"""
Experiment 3: QAT (Quantized) Voyage + Reranker
Embed (QAT) -> Atlas retrieve top-N (qat index) -> rerank -> evaluate + latency stats
"""

import json, os, time, yaml
import numpy as np
import voyageai
from utils.atlas_client import AtlasVectorSearchClient
from utils.evaluation import EvaluationMetrics
from utils.logging_utils import ExperimentLogger

def load_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def trunc(s, n):
    return s if s is None or len(s) <= n else s[:n]

def main(cfg_path="config/experiment_config.yaml", atlas_path="config/atlas_config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    atlas_cfg = yaml.safe_load(open(atlas_path))

    exp = cfg["experiments"]["experiment_3_qat_rerank"]
    emb_cfg = cfg["models"]["voyage_qat"]      # <- QAT model
    rr_cfg  = cfg["reranker"]                  # <- same reranker as exp2

    api_key = os.getenv(emb_cfg["api_key_env"])
    if not api_key:
        raise SystemExit(f"Missing env var {emb_cfg['api_key_env']}")

    vc = voyageai.Client(api_key=api_key)
    atlas = AtlasVectorSearchClient(atlas_cfg)
    evalr = EvaluationMetrics()
    log = ExperimentLogger("experiment_3_qat_rerank", cfg)

    queries = {r["query_id"]: r["query_text"] for r in load_jsonl(cfg["data"]["queries_file"])}
    qrels = {}
    for r in load_jsonl(cfg["data"]["qrels_file"]):
        qrels.setdefault(r["query_id"], {})[r["doc_id"]] = r["grade"]

    vs = exp["vector_search"]
    index, N, num_cand = vs["index_name"], vs["limit"], vs["num_candidates"]  # retrieve N candidates
    K = exp["reranker"]["final_limit"]                                       # keep top K after rerank

    all_results = {}
    total_lat, emb_lat, rr_lat = [], [], []

    for i, (qid, qtext) in enumerate(queries.items(), 1):
        t0 = time.time()

        # embed (QAT)
        t_emb0 = time.time()
        qvec = vc.embed([qtext], model=emb_cfg["model_name"], input_type="query").embeddings[0]
        e_lat = time.time() - t_emb0

        # retrieve top-N from QAT index
        retrieved, s_lat = atlas.vector_search(
            query_vector=qvec, index_name=index, limit=N, num_candidates=num_cand
        )

        # rerank to top-K (fallback to original order)
        t_rr0 = time.time()
        try:
            docs = [trunc(d.get("text", ""), rr_cfg["max_document_length"]) for d in retrieved]
            rq = trunc(qtext, rr_cfg["max_query_length"])
            rr = vc.rerank(query=rq, documents=docs, model=rr_cfg["model_name"], top_k=K)

            reranked = []
            for r in rr.results:
                d = dict(retrieved[r.index])
                d["rerank_score"] = r.relevance_score
                d["original_rank"] = r.index + 1
                reranked.append(d)
        except Exception as e:
            log.log_error("reranking_failure", str(e), {"query_id": qid, "num_docs": len(retrieved)})
            reranked = retrieved[:K]
        r_lat = time.time() - t_rr0

        t_lat = time.time() - t0
        all_results[qid] = reranked
        total_lat.append(t_lat); emb_lat.append(e_lat); rr_lat.append(r_lat)

        log.log_query_execution(
            query_id=qid,
            query_text=qtext,
            retrieval_results=retrieved,
            reranked_results=reranked,
            embedding_latency=e_lat,
            search_latency=s_lat,
            rerank_latency=r_lat,
            total_latency=t_lat,
            model_info={
                "embedding_model": f"{emb_cfg['model_name']} (QAT)",
                "reranker_model": rr_cfg["model_name"],
                "vector_index": index,
                "N_retrieved": N,
                "K_final": K,
            },
        )

        if i % 10 == 0:
            print(f"Progress: {i}/{len(queries)}")

    metrics = evalr.calculate_all_metrics(qrels, all_results)
    metrics.update({
        "mean_latency_ms": round(float(np.mean(total_lat)) * 1000, 2),
        "median_latency_ms": round(float(np.median(total_lat)) * 1000, 2),
        "p95_latency_ms": round(float(np.percentile(total_lat, 95)) * 1000, 2),
        "mean_embedding_latency_ms": round(float(np.mean(emb_lat)) * 1000, 2),
        "p95_embedding_latency_ms": round(float(np.percentile(emb_lat, 95)) * 1000, 2),
        "mean_rerank_latency_ms": round(float(np.mean(rr_lat)) * 1000, 2),
        "p95_rerank_latency_ms": round(float(np.percentile(rr_lat, 95)) * 1000, 2),
    })

    out = log.save_results(metrics, {"per_query_results": all_results})
    print(f"✅ Done. nDCG@10={metrics.get('ndcg@10', 0):.4f}  Recall@50={metrics.get('recall@50', 0):.4f}")
    print(f"   P95 total={metrics['p95_latency_ms']:.1f}ms | P95 embed(QAT)={metrics['p95_embedding_latency_ms']:.1f}ms | P95 rerank={metrics['p95_rerank_latency_ms']:.1f}ms")
    print(f"Results: {out}")
    atlas.close()

if __name__ == "__main__":
    main()
