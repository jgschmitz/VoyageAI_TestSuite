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

def main(cfg_path="config/experiment_config.yaml", atlas_path="config/atlas_config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    atlas_cfg = yaml.safe_load(open(atlas_path))

    exp = cfg["experiments"]["experiment_1_best_direct"]
    model = cfg["models"]["voyage_best"]
    api_key = os.getenv(model["api_key_env"])
    if not api_key:
        raise SystemExit(f"Missing env var {model['api_key_env']}")

    vc = voyageai.Client(api_key=api_key)
    atlas = AtlasVectorSearchClient(atlas_cfg)
    evalr = EvaluationMetrics()
    log = ExperimentLogger("experiment_1_best_direct", cfg)

    queries = {r["query_id"]: r["query_text"] for r in load_jsonl(cfg["data"]["queries_file"])}
    qrels = {}
    for r in load_jsonl(cfg["data"]["qrels_file"]):
        qrels.setdefault(r["query_id"], {})[r["doc_id"]] = r["grade"]

    K = exp["vector_search"]["limit"]
    num_cand = exp["vector_search"]["num_candidates"]
    index = exp["vector_search"]["index_name"]

    all_results, lat = {}, []
    for qid, qtext in queries.items():
        t0 = time.time()
        emb = vc.embed([qtext], model=model["model_name"], input_type="query").embeddings[0]
        results, search_lat = atlas.vector_search(
            query_vector=emb, index_name=index, limit=K, num_candidates=num_cand
        )
        total = time.time() - t0
        lat.append(total)

        all_results[qid] = results
        log.log_query_execution(
            query_id=qid,
            query_text=qtext,
            retrieval_results=results,
            reranked_results=None,
            embedding_latency=None,   # optionally measure separately
            search_latency=search_lat,
            rerank_latency=0.0,
            total_latency=total,
            model_info={"embedding_model": model["model_name"], "vector_index": index},
        )

    metrics = evalr.calculate_all_metrics(qrels, all_results)
    metrics.update({
        "mean_latency_ms": round(float(np.mean(lat)) * 1000, 2),
        "median_latency_ms": round(float(np.median(lat)) * 1000, 2),
        "p95_latency_ms": round(float(np.percentile(lat, 95)) * 1000, 2),
    })

    out = log.save_results(metrics, {"per_query_results": all_results})
    print(f"✅ Done. nDCG@10={metrics.get('ndcg@10', 0):.4f}  Recall@50={metrics.get('recall@50', 0):.4f}  P95={metrics['p95_latency_ms']:.1f}ms")
    print(f"Results: {out}")
    atlas.close()

if __name__ == "__main__":
    main()
