#!/usr/bin/env python3
"""
Experiment 4: Simple QAT (Quantized) Voyage Direct Output (No Reranker)

Goal: Best-case "cheap" pipeline - quantized embeddings only.
This is usually fastest/cheapest. Often the biggest quality tradeoff—good to know if it's "good enough".
"""

import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import voyageai
    from utils.atlas_client import AtlasVectorSearchClient
    from utils.evaluation import EvaluationMetrics
    from utils.logging_utils import ExperimentLogger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


class Experiment4QATDirect:
    """Experiment 4: Quantized Voyage model with direct retrieval (cheapest/fastest)."""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        """Initialize experiment with configuration."""
        self.config = self._load_config(config_path)
        self.atlas_config = self._load_atlas_config()
        
        # Get experiment-specific config
        self.exp_config = self.config["experiments"]["experiment_4_qat_direct"]
        self.model_config = self.config["models"]["voyage_qat"]
        
        # Initialize VoyageAI client
        api_key = os.getenv(self.model_config["api_key_env"])
        if not api_key:
            raise ValueError(f"Please set {self.model_config['api_key_env']} environment variable")
        
        self.voyage_client = voyageai.Client(api_key=api_key)
        
        # Initialize Atlas client
        self.atlas_client = AtlasVectorSearchClient(self.atlas_config)
        
        # Initialize evaluation and logging
        self.evaluator = EvaluationMetrics()
        self.logger = ExperimentLogger("experiment_4_qat_direct", self.config)
        
        print(f"✓ Initialized {self.exp_config['name']}")
        print(f"  Model: {self.model_config['model_name']} (QAT)")
        print(f"  Index: {self.exp_config['vector_search']['index_name']}")
        print("  🚀 Maximum speed/cost optimization mode")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_atlas_config(self) -> Dict[str, Any]:
        """Load Atlas configuration."""
        atlas_config_path = "config/atlas_config.yaml"
        with open(atlas_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
        """Load queries and relevance judgments."""
        # Load queries
        queries = {}
        queries_file = self.config["data"]["queries_file"]
        
        print(f"Loading queries from {queries_file}...")
        with open(queries_file, 'r') as f:
            for line in f:
                query_data = json.loads(line.strip())
                queries[query_data["query_id"]] = query_data["query_text"]
        
        # Load qrels
        qrels = {}
        qrels_file = self.config["data"]["qrels_file"]
        
        print(f"Loading relevance judgments from {qrels_file}...")
        with open(qrels_file, 'r') as f:
            for line in f:
                qrel_data = json.loads(line.strip())
                query_id = qrel_data["query_id"]
                doc_id = qrel_data["doc_id"]
                grade = qrel_data["grade"]
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = grade
        
        print(f"✓ Loaded {len(queries)} queries and {len(qrels)} relevance judgments")
        return queries, qrels
    
    def embed_query(self, query_text: str) -> tuple[List[float], float]:
        """Generate embedding for a query using Voyage QAT model."""
        start_time = time.time()
        
        try:
            result = self.voyage_client.embed(
                texts=[query_text],
                model=self.model_config["model_name"],
                input_type="query"
            )
            
            embedding = result.embeddings[0]
            latency = time.time() - start_time
            
            return embedding, latency
            
        except Exception as e:
            self.logger.log_error("embedding_failure", str(e), {"query_text": query_text})
            raise
    
    def search_documents(self, query_vector: List[float]) -> tuple[List[Dict], float]:
        """Search documents using Atlas Vector Search on QAT index."""
        search_config = self.exp_config["vector_search"]
        
        results, search_latency = self.atlas_client.vector_search(
            query_vector=query_vector,
            index_name=search_config["index_name"],  # vs_qat index
            limit=search_config["limit"],  # Direct limit, no reranking
            num_candidates=search_config["num_candidates"]
        )
        
        return results, search_latency
    
    def run_single_query(self, query_id: str, query_text: str) -> Dict[str, Any]:
        """Run experiment for a single query."""
        start_time = time.time()
        
        try:
            # Step 1: Embed query with QAT model
            query_embedding, embedding_latency = self.embed_query(query_text)
            
            # Step 2: Atlas Vector Search on vs_qat with limit=K
            search_results, search_latency = self.search_documents(query_embedding)
            
            # Step 3: Return top-K directly (no reranking)
            final_results = search_results
            
            total_latency = time.time() - start_time
            
            # Log query execution
            self.logger.log_query_execution(
                query_id=query_id,
                query_text=query_text,
                retrieval_results=search_results,
                reranked_results=None,  # No reranking in this experiment
                embedding_latency=embedding_latency,
                search_latency=search_latency,
                rerank_latency=0.0,
                total_latency=total_latency,
                model_info={
                    "embedding_model": f"{self.model_config['model_name']} (QAT)",
                    "vector_index": self.exp_config["vector_search"]["index_name"]
                }
            )
            
            return {
                "query_id": query_id,
                "results": final_results,
                "latency": {
                    "embedding": embedding_latency,
                    "search": search_latency,
                    "rerank": 0.0,
                    "total": total_latency
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to process query {query_id}: {str(e)}"
            self.logger.log_error("query_processing_failure", error_msg, {
                "query_id": query_id,
                "query_text": query_text
            })
            
            return {
                "query_id": query_id,
                "results": [],
                "error": error_msg,
                "latency": {"embedding": 0, "search": 0, "rerank": 0, "total": 0}
            }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        print(f"\n🚀 Starting {self.exp_config['name']}")
        print(f"   {self.exp_config['description']}")
        
        # Load data
        queries, qrels = self.load_data()
        
        # Run warmup queries if configured
        warmup_count = self.config["global"]["evaluation"]["warmup_queries"]
        if warmup_count > 0:
            print(f"\n🔥 Running {warmup_count} warmup queries...")
            warmup_queries = list(queries.items())[:warmup_count]
            for query_id, query_text in warmup_queries:
                self.run_single_query(query_id, query_text)
            print("✓ Warmup completed")
        
        # Run actual experiment
        print(f"\n📊 Processing {len(queries)} queries...")
        all_results = {}
        latencies = []
        embedding_latencies = []
        search_latencies = []
        
        for i, (query_id, query_text) in enumerate(queries.items(), 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(queries)} queries")
            
            result = self.run_single_query(query_id, query_text)
            all_results[query_id] = result["results"]
            latencies.append(result["latency"]["total"])
            embedding_latencies.append(result["latency"]["embedding"])
            search_latencies.append(result["latency"]["search"])
        
        print("✓ All queries processed")
        
        # Calculate evaluation metrics
        print("\n📈 Calculating evaluation metrics...")
        metrics = self.evaluator.calculate_all_metrics(qrels, all_results)
        
        # Add performance metrics (critical for this fastest/cheapest configuration)
        import numpy as np
        metrics.update({
            "mean_latency_ms": round(np.mean(latencies) * 1000, 2),
            "median_latency_ms": round(np.median(latencies) * 1000, 2),
            "p95_latency_ms": round(np.percentile(latencies, 95) * 1000, 2),
            "min_latency_ms": round(np.min(latencies) * 1000, 2),
            "max_latency_ms": round(np.max(latencies) * 1000, 2),
            "mean_embedding_latency_ms": round(np.mean(embedding_latencies) * 1000, 2),
            "p95_embedding_latency_ms": round(np.percentile(embedding_latencies, 95) * 1000, 2),
            "mean_search_latency_ms": round(np.mean(search_latencies) * 1000, 2),
            "p95_search_latency_ms": round(np.percentile(search_latencies, 95) * 1000, 2)
        })
        
        # Calculate cost efficiency (queries per second)
        total_time = sum(latencies)
        queries_per_second = len(queries) / total_time if total_time > 0 else 0
        metrics["queries_per_second"] = round(queries_per_second, 2)
        
        # Log performance stats
        self.logger.log_performance_stats("experiment_complete", {
            "total_queries": len(queries),
            "mean_latency_ms": metrics["mean_latency_ms"],
            "p95_latency_ms": metrics["p95_latency_ms"],
            "min_latency_ms": metrics["min_latency_ms"],
            "queries_per_second": metrics["queries_per_second"],
            "mean_embedding_latency_ms": metrics["mean_embedding_latency_ms"],
            "mean_search_latency_ms": metrics["mean_search_latency_ms"]
        })
        
        # Save results
        results_file = self.logger.save_results(metrics, {"per_query_results": all_results})
        
        # Print summary with focus on speed and cost efficiency
        print(f"\n✅ Experiment 4 Complete!")
        print(f"   nDCG@10: {metrics.get('ndcg@10', 0):.4f}")
        print(f"   Recall@50: {metrics.get('recall@50', 0):.4f}")
        print(f"   P95 Latency: {metrics['p95_latency_ms']:.1f}ms ⚡ (fastest)")
        print(f"   Min Latency: {metrics['min_latency_ms']:.1f}ms")
        print(f"   Queries/Second: {metrics['queries_per_second']:.1f} 🚀")
        print(f"   P95 Embedding: {metrics['p95_embedding_latency_ms']:.1f}ms (QAT)")
        print(f"   P95 Search: {metrics['p95_search_latency_ms']:.1f}ms")
        print(f"   Results saved to: {results_file}")
        
        # Provide guidance based on results
        recall_50 = metrics.get('recall@50', 0)
        if recall_50 < 0.7:
            print(f"\n⚠️  WARNING: Recall@50 is low ({recall_50:.3f})")
            print("   Consider using this configuration only for:")
            print("   - High-volume, low-precision use cases")
            print("   - First-pass filtering before more expensive methods")
            print("   - Speed-critical applications where some quality loss is acceptable")
        else:
            print(f"\n✅ Good Recall@50 ({recall_50:.3f}) - This QAT direct config looks viable!")
        
        return metrics


def main():
    """Main execution function."""
    try:
        experiment = Experiment4QATDirect()
        metrics = experiment.run_experiment()
        
        # Close connections
        experiment.atlas_client.close()
        
        return metrics
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
