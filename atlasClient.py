"""MongoDB Atlas Vector Search client."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtlasConfig:
    connection_string: str
    database_name: str = "voyageai_test_suite"
    collection_name: str = "documents"
    max_pool_size: int = 50
    min_pool_size: int = 5
    timeout_ms: int = 30000
    retry_writes: bool = True
    retry_reads: bool = True
    indexes: Dict[str, str] | None = None


@dataclass(frozen=True)
class VectorFieldConfig:
    doc_id: str
    text: str
    metadata: str
    embedding_fields_by_index: Dict[str, str]


class AtlasVectorSearchClient:
    """Client for MongoDB Atlas Vector Search operations."""

    def __init__(self, config: Dict[str, Any]):
        self.raw_config = config
        self.atlas = self._parse_atlas_config(config)
        self.fields = self._parse_vector_config(config)

        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        self.collection: Optional[Collection] = None

        self.connect()

    @staticmethod
    def _parse_atlas_config(config: Dict[str, Any]) -> AtlasConfig:
        atlas = config.get("atlas", {})
        connection_string = atlas.get("connection_string")

        if not connection_string or "YOUR_USERNAME" in connection_string:
            raise ValueError("Please configure your Atlas connection string.")

        return AtlasConfig(
            connection_string=connection_string,
            database_name=atlas.get("database_name", "voyageai_test_suite"),
            collection_name=atlas.get("collection_name", "documents"),
            max_pool_size=atlas.get("max_pool_size", 50),
            min_pool_size=atlas.get("min_pool_size", 5),
            timeout_ms=atlas.get("timeout_ms", 30000),
            retry_writes=atlas.get("retry_writes", True),
            retry_reads=atlas.get("retry_reads", True),
            indexes=atlas.get("indexes", {}),
        )

    @staticmethod
    def _parse_vector_config(config: Dict[str, Any]) -> VectorFieldConfig:
        vector_config = config.get("vector_search", {})
        fields = vector_config.get("fields", {})

        doc_id = fields.get("doc_id")
        text = fields.get("text")
        metadata = fields.get("metadata")

        if not all([doc_id, text, metadata]):
            raise ValueError("Missing required vector_search.fields configuration.")

        atlas_indexes = config.get("atlas", {}).get("indexes", {})
        embedding_fields_by_index: Dict[str, str] = {}

        best_index = atlas_indexes.get("best_model")
        qat_index = atlas_indexes.get("qat_model")

        if best_index:
            embedding_fields_by_index[best_index] = fields.get(
                "embedding_best", "embedding_best"
            )
        if qat_index:
            embedding_fields_by_index[qat_index] = fields.get(
                "embedding_qat", "embedding_qat"
            )

        return VectorFieldConfig(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            embedding_fields_by_index=embedding_fields_by_index,
        )

    def connect(self) -> None:
        """Establish connection to MongoDB Atlas."""
        try:
            self.client = MongoClient(
                self.atlas.connection_string,
                maxPoolSize=self.atlas.max_pool_size,
                minPoolSize=self.atlas.min_pool_size,
                serverSelectionTimeoutMS=self.atlas.timeout_ms,
                retryWrites=self.atlas.retry_writes,
                retryReads=self.atlas.retry_reads,
            )

            self.client.admin.command("ping")

            self.database = self.client[self.atlas.database_name]
            self.collection = self.database[self.atlas.collection_name]

            logger.info(
                "Connected to Atlas database: %s.%s",
                self.atlas.database_name,
                self.atlas.collection_name,
            )
        except ConnectionFailure:
            logger.exception("Failed to connect to Atlas")
            raise

    def _require_collection(self) -> Collection:
        if self.collection is None:
            raise RuntimeError("MongoDB collection is not initialized.")
        return self.collection

    def _require_database(self) -> Database:
        if self.database is None:
            raise RuntimeError("MongoDB database is not initialized.")
        return self.database

    def _embedding_field_for_index(self, index_name: str) -> str:
        try:
            return self.fields.embedding_fields_by_index[index_name]
        except KeyError as exc:
            raise ValueError(f"Unknown index name: {index_name}") from exc

    def vector_search(
        self,
        query_vector: List[float],
        index_name: str,
        limit: int = 10,
        num_candidates: int = 200,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Run Atlas vector search and return results plus latency."""
        collection = self._require_collection()
        start_time = time.perf_counter()

        vector_search_stage: Dict[str, Any] = {
            "$vectorSearch": {
                "index": index_name,
                "path": self._embedding_field_for_index(index_name),
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        }

        if filter_dict:
            vector_search_stage["$vectorSearch"]["filter"] = filter_dict

        pipeline = [
            vector_search_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {
                "$project": {
                    "_id": 0,
                    "doc_id": f"${self.fields.doc_id}",
                    "text": f"${self.fields.text}",
                    "score": 1,
                    "metadata": f"${self.fields.metadata}",
                }
            },
        ]

        try:
            results = list(collection.aggregate(pipeline))
            latency = time.perf_counter() - start_time
            logger.debug(
                "Vector search returned %s results in %.3fs",
                len(results),
                latency,
            )
            return results, latency
        except OperationFailure:
            logger.exception("Vector search failed")
            raise

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return a document by logical document id."""
        collection = self._require_collection()

        try:
            return collection.find_one(
                {self.fields.doc_id: doc_id},
                {"_id": 0},
            )
        except Exception:
            logger.exception("Error retrieving document %s", doc_id)
            raise

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Return documents in the same order as the input ids."""
        collection = self._require_collection()

        try:
            results = list(
                collection.find(
                    {self.fields.doc_id: {"$in": doc_ids}},
                    {"_id": 0},
                )
            )
            by_id = {doc[self.fields.doc_id]: doc for doc in results}
            return [by_id[doc_id] for doc_id in doc_ids if doc_id in by_id]
        except Exception:
            logger.exception("Error retrieving documents")
            raise

    def check_indexes(self) -> Dict[str, bool]:
        """Check whether configured Atlas search indexes exist."""
        collection = self._require_collection()

        try:
            existing = {
                idx.get("name", "")
                for idx in collection.list_search_indexes()
            }
            configured = self.atlas.indexes or {}
            return {
                index_name: index_name in existing
                for index_name in configured.values()
            }
        except Exception:
            logger.exception("Error checking indexes")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic collection statistics."""
        database = self._require_database()
        collection = self._require_collection()

        try:
            stats = database.command("collStats", collection.name)
            return {
                "document_count": stats.get("count", 0),
                "size_bytes": stats.get("size", 0),
                "average_object_size": stats.get("avgObjSize", 0),
                "index_count": stats.get("nindexes", 0),
            }
        except Exception:
            logger.exception("Error getting collection stats")
            raise

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client is not None:
            self.client.close()
            logger.info("Closed Atlas connection")

    def __enter__(self) -> "AtlasVectorSearchClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
