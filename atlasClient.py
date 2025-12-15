"""MongoDB Atlas Vector Search client for VoyageAI experiments."""

import time
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
except ImportError:
    raise ImportError("pymongo is required. Install with: pip install pymongo")


logger = logging.getLogger(__name__)


class AtlasVectorSearchClient:
    """Client for MongoDB Atlas Vector Search operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Atlas client.
        
        Args:
            config: Atlas configuration dictionary from YAML config
        """
        self.config = config
        self.atlas_config = config.get("atlas", {})
        self.vector_config = config.get("vector_search", {})
        
        # Initialize MongoDB client
        self.connection_string = self.atlas_config.get("connection_string")
        if not self.connection_string or "YOUR_USERNAME" in self.connection_string:
            raise ValueError(
                "Please configure your Atlas connection string in atlas_config.yaml"
            )
            
        self.client = None
        self.database = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB Atlas."""
        try:
            self.client = MongoClient(
                self.connection_string,
                maxPoolSize=self.atlas_config.get("max_pool_size", 50),
                minPoolSize=self.atlas_config.get("min_pool_size", 5),
                serverSelectionTimeoutMS=self.atlas_config.get("timeout_ms", 30000),
                retryWrites=self.atlas_config.get("retry_writes", True),
                retryReads=self.atlas_config.get("retry_reads", True),
            )
            
            # Test connection
            self.client.admin.command('ismaster')
            
            db_name = self.atlas_config.get("database_name", "voyageai_test_suite")
            collection_name = self.atlas_config.get("collection_name", "documents")
            
            self.database = self.client[db_name]
            self.collection = self.database[collection_name]
            
            logger.info(f"Connected to Atlas database: {db_name}.{collection_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to Atlas: {e}")
            raise
    
    def vector_search(
        self,
        query_vector: List[float],
        index_name: str,
        limit: int = 10,
        num_candidates: int = 200,
        filter_dict: Optional[Dict] = None
    ) -> Tuple[List[Dict], float]:
        """Perform vector search using Atlas Vector Search.
        
        Args:
            query_vector: The query embedding vector
            index_name: Name of the vector search index to use
            limit: Number of results to return
            num_candidates: Number of candidates to consider
            filter_dict: Optional metadata filter
            
        Returns:
            Tuple of (search results, search latency in seconds)
        """
        start_time = time.time()
        
        try:
            # Build vector search pipeline
            vector_search_stage = {
                "$vectorSearch": {
                    "index": index_name,
                    "path": self._get_embedding_field(index_name),
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit
                }
            }
            
            # Add filter if provided
            if filter_dict:
                vector_search_stage["$vectorSearch"]["filter"] = filter_dict
            
            # Build aggregation pipeline
            pipeline = [
                vector_search_stage,
                {
                    "$addFields": {
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "doc_id": f"${self.vector_config['fields']['doc_id']}",
                        "text": f"${self.vector_config['fields']['text']}",
                        "score": 1,
                        "metadata": f"${self.vector_config['fields']['metadata']}"
                    }
                }
            ]
            
            # Execute search
            results = list(self.collection.aggregate(pipeline))
            search_latency = time.time() - start_time
            
            logger.debug(f"Vector search returned {len(results)} results in {search_latency:.3f}s")
            
            return results, search_latency
            
        except OperationFailure as e:
            logger.error(f"Vector search failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during vector search: {e}")
            raise
    
    def _get_embedding_field(self, index_name: str) -> str:
        """Get the embedding field name for a given index."""
        indexes = self.atlas_config.get("indexes", {})
        fields = self.vector_config.get("fields", {})
        
        if index_name == indexes.get("best_model"):
            return fields.get("embedding_best", "embedding_best")
        elif index_name == indexes.get("qat_model"):
            return fields.get("embedding_qat", "embedding_qat")
        else:
            raise ValueError(f"Unknown index name: {index_name}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by its ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
        """
        try:
            doc_id_field = self.vector_config["fields"]["doc_id"]
            result = self.collection.find_one(
                {doc_id_field: doc_id},
                {"_id": 0}  # Exclude MongoDB ObjectId
            )
            return result
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict]:
        """Retrieve multiple documents by their IDs.
        
        Args:
            doc_ids: List of document identifiers
            
        Returns:
            List of document dictionaries
        """
        try:
            doc_id_field = self.vector_config["fields"]["doc_id"]
            results = list(self.collection.find(
                {doc_id_field: {"$in": doc_ids}},
                {"_id": 0}  # Exclude MongoDB ObjectId
            ))
            
            # Maintain order of input doc_ids
            doc_map = {doc[doc_id_field]: doc for doc in results}
            ordered_results = [doc_map.get(doc_id) for doc_id in doc_ids]
            
            # Filter out None values (missing documents)
            return [doc for doc in ordered_results if doc is not None]
            
        except Exception as e:
            logger.error(f"Error retrieving documents {doc_ids}: {e}")
            return []
    
    def check_indexes(self) -> Dict[str, bool]:
        """Check if required vector search indexes exist.
        
        Returns:
            Dictionary with index names and their existence status
        """
        try:
            # Get list of search indexes
            indexes = list(self.collection.list_search_indexes())
            index_names = [idx.get("name", "") for idx in indexes]
            
            required_indexes = self.atlas_config.get("indexes", {})
            status = {}
            
            for index_key, index_name in required_indexes.items():
                status[index_name] = index_name in index_names
                
            logger.info(f"Index status: {status}")
            return status
            
        except Exception as e:
            logger.error(f"Error checking indexes: {e}")
            return {}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = self.database.command("collStats", self.collection.name)
            
            return {
                "document_count": stats.get("count", 0),
                "size_bytes": stats.get("size", 0),
                "average_object_size": stats.get("avgObjSize", 0),
                "index_count": stats.get("nindexes", 0),
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Closed Atlas connection")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
