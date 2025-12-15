"""Evaluation metrics for retrieval experiments."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)


Qrels = Mapping[str, Mapping[str, int]]                 # {qid: {doc_id: grade}}
Results = Mapping[str, Sequence[Mapping[str, Any]]]     # {qid: [{"doc_id":..., ...}, ...]}


def _doc_ids(results: Sequence[Mapping[str, Any]]) -> List[str]:
    return [r["doc_id"] for r in results if "doc_id" in r]


def _relevant_set(query_qrels: Mapping[str, int]) -> set[str]:
    # IMPORTANT: only grades > 0 are relevant
    return {doc_id for doc_id, grade in query_qrels.items() if grade > 0}


def _dcg(grades: List[int]) -> float:
    # grades are in rank order, already truncated to k
    s = 0.0
    for i, rel in enumerate(grades):
        if rel > 0:
            s += (2**rel - 1) / math.log2(i + 2)  # i is 0-indexed
    return s


def _ndcg_at_k(query_qrels: Mapping[str, int], retrieved: List[str], k: int) -> float:
    retrieved = retrieved[:k]
    gains = [query_qrels.get(d, 0) for d in retrieved]
    dcg = _dcg(gains)

    ideal = sorted([g for g in query_qrels.values() if g > 0], reverse=True)[:k]
    idcg = _dcg(ideal)
    return (dcg / idcg) if idcg > 0 else 0.0


def _precision_at_k(rel: set[str], retrieved: List[str], k: int) -> float:
    retrieved = retrieved[:k]
    if not retrieved:
        return 0.0
    return sum(1 for d in retrieved if d in rel) / len(retrieved)


def _recall_at_k(rel: set[str], retrieved: List[str], k: int) -> float:
    if not rel:
        return 0.0
    retrieved = retrieved[:k]
    return sum(1 for d in retrieved if d in rel) / len(rel)


def _mrr(rel: set[str], retrieved: List[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def _ap(rel: set[str], retrieved: List[str]) -> float:
    if not rel:
        return 0.0
    hit = 0
    s = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            hit += 1
            s += hit / i
    return s / len(rel)


@dataclass(frozen=True)
class CoverageStats:
    query_coverage: float
    document_coverage: float
    total_queries: int
    queries_with_results: int
    total_relevant_docs: int
    unique_retrieved_docs: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "query_coverage": self.query_coverage,
            "document_coverage": self.document_coverage,
            "total_queries": float(self.total_queries),
            "queries_with_results": float(self.queries_with_results),
            "total_relevant_docs": float(self.total_relevant_docs),
            "unique_retrieved_docs": float(self.unique_retrieved_docs),
        }


class EvaluationMetrics:
    """Compute retrieval metrics (macro-averaged across queries)."""

    def calculate_all_metrics(
        self,
        qrels: Qrels,
        results: Results,
        k_values: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        k_values = list(k_values) if k_values is not None else [5, 10, 20, 50]
        metrics: Dict[str, float] = {}

        for k in k_values:
            metrics[f"precision@{k}"] = self._macro(qrels, results, lambda qq, rr: _precision_at_k(_relevant_set(qq), rr, k))
            metrics[f"recall@{k}"] = self._macro(qrels, results, lambda qq, rr: _recall_at_k(_relevant_set(qq), rr, k))
            metrics[f"ndcg@{k}"] = self._macro(qrels, results, lambda qq, rr: _ndcg_at_k(qq, rr, k))

        metrics["mrr"] = self._macro(qrels, results, lambda qq, rr: _mrr(_relevant_set(qq), rr))
        metrics["map"] = self._macro(qrels, results, lambda qq, rr: _ap(_relevant_set(qq), rr))

        metrics.update(self.calculate_coverage(qrels, results).as_dict())
        return metrics

    def calculate_coverage(self, qrels: Qrels, results: Results) -> CoverageStats:
        total_queries = len(qrels)
        queries_with_results = sum(1 for qid in qrels.keys() if qid in results and len(results[qid]) > 0)
        query_coverage = (queries_with_results / total_queries) if total_queries else 0.0

        all_rel = set()
        for qq in qrels.values():
            all_rel |= _relevant_set(qq)

        retrieved = set()
        for qid, res in results.items():
            if qid in qrels:
                retrieved |= set(_doc_ids(res))

        doc_coverage = (len(all_rel & retrieved) / len(all_rel)) if all_rel else 0.0
        return CoverageStats(
            query_coverage=query_coverage,
            document_coverage=doc_coverage,
            total_queries=total_queries,
            queries_with_results=queries_with_results,
            total_relevant_docs=len(all_rel),
            unique_retrieved_docs=len(retrieved),
        )

    def compare_experiments(self, baseline: Dict[str, float], experiment: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        out = {}
        for m, b in baseline.items():
            if m not in experiment:
                continue
            e = experiment[m]
            abs_diff = e - b
            rel_diff = (abs_diff / b * 100.0) if b != 0 else 0.0
            out[m] = {
                "baseline": b,
                "experiment": e,
                "absolute_difference": abs_diff,
                "relative_difference_percent": rel_diff,
                "improvement": float(abs_diff > 0),
            }
        return out

    def calculate_statistical_significance(
        self,
        qrels: Qrels,
        results_a: Results,
        results_b: Results,
        metric: str = "ndcg@10",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        try:
            from scipy.stats import ttest_rel
        except ImportError:
            logger.warning("scipy not available, skipping significance test")
            return {"error": "scipy required for significance testing"}

        per_a, per_b = [], []
        common = set(results_a.keys()) & set(results_b.keys()) & set(qrels.keys())

        scorer = self._metric_fn(metric)
        for qid in common:
            ra = _doc_ids(results_a[qid])
            rb = _doc_ids(results_b[qid])
            per_a.append(scorer(qrels[qid], ra))
            per_b.append(scorer(qrels[qid], rb))

        if len(per_a) < 10:
            return {"error": f"Not enough queries ({len(per_a)}) for reliable significance testing"}

        stat, p = ttest_rel(per_a, per_b)
        return {
            "metric": metric,
            "num_queries": len(per_a),
            "mean_a": float(np.mean(per_a)),
            "mean_b": float(np.mean(per_b)),
            "statistic": float(stat),
            "p_value": float(p),
            "significant": bool(p < alpha),
            "alpha": alpha,
        }

    def _macro(self, qrels: Qrels, results: Results, fn) -> float:
        vals = []
        for qid, qq in qrels.items():
            rr = _doc_ids(results.get(qid, []))
            vals.append(fn(qq, rr))
        return float(np.mean(vals)) if vals else 0.0

    def _metric_fn(self, metric: str):
        if metric.startswith("ndcg@"):
            k = int(metric.split("@")[1])
            return lambda qq, rr: _ndcg_at_k(qq, rr, k)
        if metric.startswith("precision@"):
            k = int(metric.split("@")[1])
            return lambda qq, rr: _precision_at_k(_relevant_set(qq), rr, k)
        if metric.startswith("recall@"):
            k = int(metric.split("@")[1])
            return lambda qq, rr: _recall_at_k(_relevant_set(qq), rr, k)
        raise ValueError(f"Unsupported metric: {metric}")
