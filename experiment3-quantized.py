#!/usr/bin/env python3
"""
Experiment 3: QAT (Quantized) Voyage + Rescorer Output

Goal: See if quantizing the embedding model reduces cost/latency while maintaining quality 
(with reranker safety net). Reranker often masks small embedding degradation.
If QAT retriever drops Recall@50 a lot, reranker can't recover missing docs.
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


class Experiment3QATRerank:
    """Experiment 3: Quantized Voyage model with reranker safety net."""
    
    def __init__(self, config_path: str = "config/experiment_config.yaml"):
        """Initialize experiment with configuration."""
        self.config = self._load_config(config_path)
        self.atlas_config = self._load_atlas_config()
        
        # Get experiment-specific config
        self.exp_config = self.config["experiments"]["experiment_3_qat_rerank"]
        self.model_config = self.config["models"]["voyage_qat"]
        self.rerank_config = self.config["reranker"]
        
        # Initialize VoyageAI client
        api_key = os.getenv(self.model_config["api_key_env"])
        if not api_key:
            raise ValueError(f"Please set {self.model_config['api_key_env']} environment variable")
        
        self.voyage_client = voyageai.Client(api_key=api_key)
        
        # Initialize Atlas client
        self.atlas_client = AtlasVectorSearchClient(self.atlas_config)
        
        # Initialize evaluation and logging
        self.evaluator = EvaluationMetrics()
        self.logger = ExperimentLogger("experiment_3_qat_rerank", self.config)
        
        print(f"✓ Initialized {self.exp_config['name']}")
        print(f"  Embedding Model: {self.model_config['model_name']} (QAT)")
        print(f"  Reranker Model: {self.rerank_config['model_name']}")
        print(f"  Index: {self.exp_config['vector_search']['index_name']}")
    
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
            limit=search_config["limit"],  # Retrieve more candidates for reranking
            num_candidates=search_config["num_candidates"]
        )
        
        return results, search_latency
    
    def rerank_documents(self, query_text: str, documents: List[Dict]) -> tuple[List[Dict], float]:
        """Rerank documents using Voyage Reranker."""
        if not documents:
            return [], 0.0
            
        start_time = time.time()
        
        try:
            # Prepare documents for reranking
            doc_texts = []
            for doc in documents:
                # Use the text field from the document
                doc_text = doc.get("text", "")
                # Truncate if too long
                max_length = self.rerank_config["max_document_length"]
                if len(doc_text) > max_length:
                    doc_text = doc_text[:max_length]
                doc_texts.append(doc_text)
            
            # Truncate query if needed
            query_text_truncated = query_text
            max_query_length = self.rerank_config["max_query_length"]
            if len(query_text) > max_query_length:
                query_text_truncated = query_text[:max_query_length]
            
            # Call Voyage Reranker
            rerank_result = self.voyage_client.rerank(
                query=query_text_truncated,
                documents=doc_texts,
                model=self.rerank_config["model_name"],
                top_k=self.exp_config["reranker"]["final_limit"]
            )
            
            # Reconstruct results with rerank scores
            reranked_results = []
            for result in rerank_result.results:
                original_doc = documents[result.index]
                reranked_doc = original_doc.copy()
                reranked_doc["rerank_score"] = result.relevance_score
                reranked_doc["original_rank"] = result.index + 1
                reranked_results.append(reranked_doc)
            
            rerank_latency = time.time() - start_time
            
            return reranked_results, rerank_latency
            
        except Exception as e:
            self.logger.log_error("reranking_failure", str(e), {
                "query_text": query_text,
                "num_documents": len(documents)
            })
            # Fallback to original ranking
            final_limit = self.exp_config["reranker"]["final_limit"]
            return documents[:final_limit], time.time() - start_time
    
    def run_single_query(self, query_id: str, query_text: str) -> Dict[str, Any]:
        """Run experiment for a single query."""
        start_time = time.time()
        
        try:
            # Step 1: Embed query with quantized/QAT Voyage model
            query_embedding, embedding_latency = self.embed_query(query_text)
            
            # Step 2: Atlas Vector Search on vs_qat retrieving top-N candidates
            # Keep N, numCandidates identical to Experiment 2
            search_results, search_latency = self.search_documents(query_embedding)
            
            # Step 3: Rerank the same way with the Rescorer
            reranked_results, rerank_latency = self.rerank_documents(query_text, search_results)
            
            # Step 4: Return top-K after rerank
            final_results = reranked_results
            
            total_latency = time.time() - start_time
            
            # Log query execution
            self.logger.log_query_execution(
                query_id=query_id,
                query_text=query_text,
                retrieval_results=search_results,
                reranked_results=reranked_results,
                embedding_latency=embedding_latency,
                search_latency=search_latency,
                rerank_latency=rerank_latency,
                total_latency=total_latency,
                model_info={
                    "embedding_model": f"{self.model_config['model_name']} (QAT)",
                    "reranker_model": self.rerank_config["model_name"],
                    "vector_index": self.exp_config["vector_search"]["index_name"]
                }
            )
            
            return {
                "query_id": query_id,
                "results": final_results,
                "latency": {
                    "embedding": embedding_latency,
                    "search": search_latency,
                    "rerank": rerank_latency,
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
        rerank_latencies = []
        
        for i, (query_id, query_text) in enumerate(queries.items(), 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(queries)} queries")
            
            result = self.run_single_query(query_id, query_text)
            all_results[query_id] = result["results"]
            latencies.append(result["latency"]["total"])
            embedding_latencies.append(result["latency"]["embedding"])
            rerank_latencies.append(result["latency"]["rerank"])
        
        print("✓ All queries processed")
        
        # Calculate evaluation metrics
        print("\n📈 Calculating evaluation metrics...")
        metrics = self.evaluator.calculate_all_metrics(qrels, all_results)
        
        # Add performance metrics (key for cost/latency analysis)
        import numpy as np
        metrics.update({
            "mean_latency_ms": round(np.mean(latencies) * 1000, 2),
            "median_latency_ms": round(np.median(latencies) * 1000, 2),
            "p95_latency_ms": round(np.percentile(latencies, 95) * 1000, 2),
            "mean_embedding_latency_ms": round(np.mean(embedding_latencies) * 1000, 2),
            "p95_embedding_latency_ms": round(np.percentile(embedding_latencies, 95) * 1000, 2),
            "mean_rerank_latency_ms": round(np.mean(rerank_latencies) * 1000, 2),
            "p95_rerank_latency_ms": round(np.percentile(rerank_latencies, 95) * 1000, 2)
        })
        
        # Log performance stats
        self.logger.log_performance_stats("experiment_complete", {
            "total_queries": len(queries),
            "mean_latency_ms": metrics["mean_latency_ms"],
            "p95_latency_ms": metrics["p95_latency_ms"],
            "mean_embedding_latency_ms": metrics["mean_embedding_latency_ms"],
            "mean_rerank_latency_ms": metrics["mean_rerank_latency_ms"]
        })
        
        # Save results
        results_file = self.logger.save_results(metrics, {"per_query_results": all_results})
        
        # Print summary with focus on cost/quality trade-off
        print(f"\n✅ Experiment 3 Complete!")
        print(f"   nDCG@10: {metrics.get('ndcg@10', 0):.4f}")
        print(f"   Recall@50: {metrics.get('recall@50', 0):.4f}")
        print(f"   P95 Total Latency: {metrics['p95_latency_ms']:.1f}ms")
        print(f"   P95 Embedding Latency: {metrics['p95_embedding_latency_ms']:.1f}ms (QAT)")
        print(f"   P95 Rerank Latency: {metrics['p95_rerank_latency_ms']:.1f}ms")
        print(f"   Results saved to: {results_file}")
        
        return metrics


def main():
    """Main execution function."""
    try:
        experiment = Experiment3QATRerank()
        metrics = experiment.run_experiment()
        
        # Close connections
        experiment.atlas_client.close()
        
        return metrics
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
