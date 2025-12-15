"""Evaluation metrics for VoyageAI experiments."""

import math
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculate evaluation metrics for retrieval experiments."""
    
    def __init__(self):
        """Initialize the evaluation metrics calculator."""
        self.metrics_cache = {}
    
    def calculate_all_metrics(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, List[Dict[str, Any]]],
        k_values: List[int] = [5, 10, 20, 50]
    ) -> Dict[str, float]:
        """Calculate all standard retrieval metrics.
        
        Args:
            qrels: Relevance judgments {query_id: {doc_id: grade}}
            results: Search results {query_id: [{'doc_id': str, 'score': float}, ...]}
            k_values: List of k values for precision@k, recall@k, ndcg@k
            
        Returns:
            Dictionary of metric names to values
        """
        all_metrics = {}
        
        # Calculate metrics for each k value
        for k in k_values:
            precision_k = self.precision_at_k(qrels, results, k)
            recall_k = self.recall_at_k(qrels, results, k)
            ndcg_k = self.ndcg_at_k(qrels, results, k)
            
            all_metrics[f"precision@{k}"] = precision_k
            all_metrics[f"recall@{k}"] = recall_k
            all_metrics[f"ndcg@{k}"] = ndcg_k
        
        # Calculate ranking metrics
        all_metrics["mrr"] = self.mean_reciprocal_rank(qrels, results)
        all_metrics["map"] = self.mean_average_precision(qrels, results)
        
        # Calculate coverage metrics
        coverage_stats = self.calculate_coverage(qrels, results)
        all_metrics.update(coverage_stats)
        
        return all_metrics
    
    def precision_at_k(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, List[Dict[str, Any]]],
        k: int
    ) -> float:
        """Calculate Precision@K.
        
        Args:
            qrels: Relevance judgments
            results: Search results
            k: Cut-off value
            
        Returns:
            Precision@K score
        """
        precisions = []
        
        for query_id, query_results in results.items():
            if query_id not in qrels:
                continue
                
            relevant_docs = set(qrels[query_id].keys())
            retrieved_docs = [doc["doc_id"] for doc in query_results[:k]]
            
            if not retrieved_docs:
                precisions.append(0.0)
                continue
            
            relevant_retrieved = len([doc_id for doc_id in retrieved_docs if doc_id in relevant_docs])
            precision = relevant_retrieved / len(retrieved_docs)
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, List[Dict[str, Any]]],
        k: int
    ) -> float:
        """Calculate Recall@K.
        
        Args:
            qrels: Relevance judgments
            results: Search results
            k: Cut-off value
            
        Returns:
            Recall@K score
        """
        recalls = []
        
        for query_id, query_results in results.items():
            if query_id not in qrels:
                continue
                
            relevant_docs = set(qrels[query_id].keys())
            retrieved_docs = [doc["doc_id"] for doc in query_results[:k]]
            
            if not relevant_docs:
                continue
            
            relevant_retrieved = len([doc_id for doc_id in retrieved_docs if doc_id in relevant_docs])
            recall = relevant_retrieved / len(relevant_docs)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def ndcg_at_k(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, List[Dict[str, Any]]],
        k: int
    ) -> float:
        """Calculate NDCG@K.
        
        Args:
            qrels: Relevance judgments (grades can be 0-3 or binary 0/1)
            results: Search results
            k: Cut-off value
            
        Returns:
            NDCG@K score
        """
        ndcg_scores = []
        
        for query_id, query_results in results.items():
            if query_id not in qrels:
                continue
                
            # Get relevance grades for this query
            query_qrels = qrels[query_id]
            retrieved_docs = [doc["doc_id"] for doc in query_results[:k]]
            
            # Calculate DCG@k
            dcg = 0.0
            for i, doc_id in enumerate(retrieved_docs):
                rel = query_qrels.get(doc_id, 0)
                if rel > 0:
                    dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because i is 0-indexed
            
            # Calculate IDCG@k (perfect ranking)
            ideal_gains = sorted(query_qrels.values(), reverse=True)[:k]
            idcg = 0.0
            for i, rel in enumerate(ideal_gains):
                if rel > 0:
                    idcg += (2**rel - 1) / math.log2(i + 2)
            
            # Calculate NDCG@k
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_scores.append(ndcg)
            else:
                ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def mean_reciprocal_rank(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, List[Dict[str, Any]]]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            qrels: Relevance judgments
            results: Search results
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for query_id, query_results in results.items():
            if query_id not in qrels:
                continue
                
            relevant_docs = set(qrels[query_id].keys())
            
            # Find first relevant document
            first_relevant_rank = None
            for i, doc in enumerate(query_results):
                if doc["doc_id"] in relevant_docs:
                    first_relevant_rank = i + 1  # 1-indexed rank
                    break
            
            if first_relevant_rank is not None:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def mean_average_precision(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, List[Dict[str, Any]]]
    ) -> float:
        """Calculate Mean Average Precision (MAP).
        
        Args:
            qrels: Relevance judgments
            results: Search results
            
        Returns:
            MAP score
        """
        average_precisions = []
        
        for query_id, query_results in results.items():
            if query_id not in qrels:
                continue
                
            relevant_docs = set(qrels[query_id].keys())
            
            if not relevant_docs:
                continue
            
            # Calculate average precision for this query
            num_relevant = 0
            sum_precision = 0.0
            
            for i, doc in enumerate(query_results):
                if doc["doc_id"] in relevant_docs:
                    num_relevant += 1
                    precision = num_relevant / (i + 1)
                    sum_precision += precision
            
            if num_relevant > 0:
                average_precision = sum_precision / len(relevant_docs)
                average_precisions.append(average_precision)
        
        return np.mean(average_precisions) if average_precisions else 0.0
    
    def calculate_coverage(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate coverage statistics.
        
        Args:
            qrels: Relevance judgments
            results: Search results
            
        Returns:
            Dictionary with coverage metrics
        """
        total_queries = len(qrels)
        queries_with_results = len([q for q in results.keys() if q in qrels])
        
        # Query coverage
        query_coverage = queries_with_results / total_queries if total_queries > 0 else 0.0
        
        # Document coverage
        all_relevant_docs = set()
        for query_qrels in qrels.values():
            all_relevant_docs.update(query_qrels.keys())
        
        retrieved_docs = set()
        for query_results in results.values():
            for doc in query_results:
                retrieved_docs.add(doc["doc_id"])
        
        relevant_retrieved = all_relevant_docs.intersection(retrieved_docs)
        doc_coverage = len(relevant_retrieved) / len(all_relevant_docs) if all_relevant_docs else 0.0
        
        return {
            "query_coverage": query_coverage,
            "document_coverage": doc_coverage,
            "total_queries": total_queries,
            "queries_with_results": queries_with_results,
            "total_relevant_docs": len(all_relevant_docs),
            "unique_retrieved_docs": len(retrieved_docs)
        }
    
    def compare_experiments(
        self,
        baseline_metrics: Dict[str, float],
        experiment_metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare two sets of metrics.
        
        Args:
            baseline_metrics: Baseline experiment metrics
            experiment_metrics: Comparison experiment metrics
            
        Returns:
            Dictionary with comparison statistics
        """
        comparison = {}
        
        for metric_name in baseline_metrics.keys():
            if metric_name in experiment_metrics:
                baseline_val = baseline_metrics[metric_name]
                experiment_val = experiment_metrics[metric_name]
                
                # Calculate absolute and relative differences
                abs_diff = experiment_val - baseline_val
                rel_diff = (abs_diff / baseline_val * 100) if baseline_val != 0 else 0
                
                comparison[metric_name] = {
                    "baseline": baseline_val,
                    "experiment": experiment_val,
                    "absolute_difference": abs_diff,
                    "relative_difference_percent": rel_diff,
                    "improvement": abs_diff > 0
                }
        
        return comparison
    
    def calculate_statistical_significance(
        self,
        qrels: Dict[str, Dict[str, int]],
        results_a: Dict[str, List[Dict[str, Any]]],
        results_b: Dict[str, List[Dict[str, Any]]],
        metric: str = "ndcg@10",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Calculate statistical significance using paired t-test.
        
        Args:
            qrels: Relevance judgments
            results_a: Results from system A
            results_b: Results from system B
            metric: Metric to test (default: ndcg@10)
            alpha: Significance level
            
        Returns:
            Dictionary with significance test results
        """
        try:
            from scipy.stats import ttest_rel
        except ImportError:
            logger.warning("scipy not available, skipping significance test")
            return {"error": "scipy required for significance testing"}
        
        # Extract metric values for each query
        scores_a = []
        scores_b = []
        
        common_queries = set(results_a.keys()).intersection(set(results_b.keys()))
        
        for query_id in common_queries:
            if query_id not in qrels:
                continue
            
            # Calculate metric for system A
            score_a = self._calculate_single_metric(qrels[query_id], results_a[query_id], metric)
            score_b = self._calculate_single_metric(qrels[query_id], results_b[query_id], metric)
            
            scores_a.append(score_a)
            scores_b.append(score_b)
        
        if len(scores_a) < 10:
            return {"error": f"Not enough queries ({len(scores_a)}) for reliable significance testing"}
        
        # Perform paired t-test
        statistic, p_value = ttest_rel(scores_a, scores_b)
        
        return {
            "metric": metric,
            "num_queries": len(scores_a),
            "mean_a": np.mean(scores_a),
            "mean_b": np.mean(scores_b),
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha
        }
    
    def _calculate_single_metric(
        self,
        query_qrels: Dict[str, int],
        query_results: List[Dict[str, Any]],
        metric: str
    ) -> float:
        """Calculate a single metric for one query."""
        if metric.startswith("ndcg@"):
            k = int(metric.split("@")[1])
            return self._single_query_ndcg(query_qrels, query_results, k)
        elif metric.startswith("precision@"):
            k = int(metric.split("@")[1])
            return self._single_query_precision(query_qrels, query_results, k)
        elif metric.startswith("recall@"):
            k = int(metric.split("@")[1])
            return self._single_query_recall(query_qrels, query_results, k)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def _single_query_ndcg(self, query_qrels: Dict[str, int], query_results: List[Dict[str, Any]], k: int) -> float:
        """Calculate NDCG@k for a single query."""
        retrieved_docs = [doc["doc_id"] for doc in query_results[:k]]
        
        # Calculate DCG@k
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs):
            rel = query_qrels.get(doc_id, 0)
            if rel > 0:
                dcg += (2**rel - 1) / math.log2(i + 2)
        
        # Calculate IDCG@k
        ideal_gains = sorted(query_qrels.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_gains):
            if rel > 0:
                idcg += (2**rel - 1) / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _single_query_precision(self, query_qrels: Dict[str, int], query_results: List[Dict[str, Any]], k: int) -> float:
        """Calculate Precision@k for a single query."""
        relevant_docs = set(query_qrels.keys())
        retrieved_docs = [doc["doc_id"] for doc in query_results[:k]]
        
        if not retrieved_docs:
            return 0.0
        
        relevant_retrieved = len([doc_id for doc_id in retrieved_docs if doc_id in relevant_docs])
        return relevant_retrieved / len(retrieved_docs)
    
    def _single_query_recall(self, query_qrels: Dict[str, int], query_results: List[Dict[str, Any]], k: int) -> float:
        """Calculate Recall@k for a single query."""
        relevant_docs = set(query_qrels.keys())
        retrieved_docs = [doc["doc_id"] for doc in query_results[:k]]
        
        if not relevant_docs:
            return 0.0
        
        relevant_retrieved = len([doc_id for doc_id in retrieved_docs if doc_id in relevant_docs])
        return relevant_retrieved / len(relevant_docs)
