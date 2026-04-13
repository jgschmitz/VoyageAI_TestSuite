"""Evaluation metrics for retrieval experiments."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

Qrels = Mapping[str, Mapping[str, int]]             # {qid: {doc_id: grade}}
Results = Mapping[str, Sequence[Mapping[str, Any]]] # {qid: [{"doc_id": ..., ...}, ...]}


def _doc_ids(results: Sequence[Mapping[str, Any]]) -> List[str]:
    return [result["doc_id"] for result in results if "doc_id" in result]


def _relevant_doc_ids(query_qrels: Mapping[str, int]) -> set[str]:
    return {doc_id for doc_id, grade in query_qrels.items() if grade > 0}


def _dcg(grades: Sequence[int]) -> float:
    score = 0.0
    for rank_index, grade in enumerate(grades):
        if grade > 0:
            score += (2**grade - 1) / math.log2(rank_index + 2)
    return score


def _ndcg_at_k(query_qrels: Mapping[str, int], retrieved_doc_ids: Sequence[str], k: int) -> float:
    top_k = list(retrieved_doc_ids[:k])
    gains = [query_qrels.get(doc_id, 0) for doc_id in top_k]
    dcg = _dcg(gains)

    ideal_gains = sorted((grade for grade in query_qrels.values() if grade > 0), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_gains)

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def _precision_at_k(relevant_doc_ids: set[str], retrieved_doc_ids: Sequence[str], k: int) -> float:
    top_k = list(retrieved_doc_ids[:k])
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / len(top_k)


def _recall_at_k(relevant_doc_ids: set[str], retrieved_doc_ids: Sequence[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    top_k = list(retrieved_doc_ids[:k])
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def _mrr(relevant_doc_ids: set[str], retrieved_doc_ids: Sequence[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def _average_precision(relevant_doc_ids: set[str], retrieved_doc_ids: Sequence[str]) -> float:
    if not relevant_doc_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant_doc_ids)


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
    """Compute macro-averaged retrieval metrics across queries."""

    def calculate_all_metrics(
        self,
        qrels: Qrels,
        results: Results,
        k_values: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        ks = list(k_values) if k_values is not None else [5, 10, 20, 50]
        metrics: Dict[str, float] = {}

        for k in ks:
            metrics[f"precision@{k}"] = self._macro_average(
                qrels,
                results,
                lambda query_qrels, retrieved: _precision_at_k(
                    _relevant_doc_ids(query_qrels), retrieved, k
                ),
            )
            metrics[f"recall@{k}"] = self._macro_average(
                qrels,
                results,
                lambda query_qrels, retrieved: _recall_at_k(
                    _relevant_doc_ids(query_qrels), retrieved, k
                ),
            )
            metrics[f"ndcg@{k}"] = self._macro_average(
                qrels,
                results,
                lambda query_qrels, retrieved: _ndcg_at_k(query_qrels, retrieved, k),
            )

        metrics["mrr"] = self._macro_average(
            qrels,
            results,
            lambda query_qrels, retrieved: _mrr(_relevant_doc_ids(query_qrels), retrieved),
        )
        metrics["map"] = self._macro_average(
            qrels,
            results,
            lambda query_qrels, retrieved: _average_precision(
                _relevant_doc_ids(query_qrels), retrieved
            ),
        )

        metrics.update(self.calculate_coverage(qrels, results).as_dict())
        return metrics

    def calculate_coverage(self, qrels: Qrels, results: Results) -> CoverageStats:
        total_queries = len(qrels)
        queries_with_results = sum(
            1 for query_id in qrels if results.get(query_id)
        )
        query_coverage = queries_with_results / total_queries if total_queries else 0.0

        all_relevant_doc_ids: set[str] = set()
        for query_qrels in qrels.values():
            all_relevant_doc_ids |= _relevant_doc_ids(query_qrels)

        retrieved_doc_ids: set[str] = set()
        for query_id in qrels:
            retrieved_doc_ids |= set(_doc_ids(results.get(query_id, [])))

        document_coverage = (
            len(all_relevant_doc_ids & retrieved_doc_ids) / len(all_relevant_doc_ids)
            if all_relevant_doc_ids
            else 0.0
        )

        return CoverageStats(
            query_coverage=query_coverage,
            document_coverage=document_coverage,
            total_queries=total_queries,
            queries_with_results=queries_with_results,
            total_relevant_docs=len(all_relevant_doc_ids),
            unique_retrieved_docs=len(retrieved_doc_ids),
        )

    def compare_experiments(
        self,
        baseline: Dict[str, float],
        experiment: Dict[str, float],
    ) -> Dict[str, Dict[str, float | bool]]:
        comparison: Dict[str, Dict[str, float | bool]] = {}

        for metric_name, baseline_value in baseline.items():
            if metric_name not in experiment:
                continue

            experiment_value = experiment[metric_name]
            absolute_difference = experiment_value - baseline_value
            relative_difference_percent = (
                (absolute_difference / baseline_value) * 100.0
                if baseline_value != 0
                else 0.0
            )

            comparison[metric_name] = {
                "baseline": baseline_value,
                "experiment": experiment_value,
                "absolute_difference": absolute_difference,
                "relative_difference_percent": relative_difference_percent,
                "improved": absolute_difference > 0,
            }

        return comparison

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

        metric_fn = self._metric_function(metric)
        common_query_ids = sorted(set(qrels) & set(results_a) & set(results_b))

        scores_a: List[float] = []
        scores_b: List[float] = []

        for query_id in common_query_ids:
            query_qrels = qrels[query_id]
            retrieved_a = _doc_ids(results_a[query_id])
            retrieved_b = _doc_ids(results_b[query_id])

            scores_a.append(metric_fn(query_qrels, retrieved_a))
            scores_b.append(metric_fn(query_qrels, retrieved_b))

        if len(scores_a) < 10:
            return {
                "error": f"Not enough queries ({len(scores_a)}) for reliable significance testing"
            }

        statistic, p_value = ttest_rel(scores_a, scores_b)

        return {
            "metric": metric,
            "num_queries": len(scores_a),
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "alpha": alpha,
        }

    def _macro_average(
        self,
        qrels: Qrels,
        results: Results,
        metric_fn: Callable[[Mapping[str, int], Sequence[str]], float],
    ) -> float:
        values: List[float] = []

        for query_id, query_qrels in qrels.items():
            retrieved_doc_ids = _doc_ids(results.get(query_id, []))
            values.append(metric_fn(query_qrels, retrieved_doc_ids))

        return float(np.mean(values)) if values else 0.0

    def _metric_function(
        self,
        metric: str,
    ) -> Callable[[Mapping[str, int], Sequence[str]], float]:
        if metric.startswith("ndcg@"):
            k = int(metric.split("@", 1)[1])
            return lambda query_qrels, retrieved: _ndcg_at_k(query_qrels, retrieved, k)

        if metric.startswith("precision@"):
            k = int(metric.split("@", 1)[1])
            return lambda query_qrels, retrieved: _precision_at_k(
                _relevant_doc_ids(query_qrels), retrieved, k
            )

        if metric.startswith("recall@"):
            k = int(metric.split("@", 1)[1])
            return lambda query_qrels, retrieved: _recall_at_k(
                _relevant_doc_ids(query_qrels), retrieved, k
            )

        if metric == "mrr":
            return lambda query_qrels, retrieved: _mrr(
                _relevant_doc_ids(query_qrels), retrieved
            )

        if metric == "map":
            return lambda query_qrels, retrieved: _average_precision(
                _relevant_doc_ids(query_qrels), retrieved
            )

        raise ValueError(f"Unsupported metric: {metric}")
