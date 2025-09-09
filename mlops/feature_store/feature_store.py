"""
Production-grade Feature Store for MongoDB Query Translator.
Manages embeddings, features, and metadata with versioning and lineage tracking.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
import mlflow
import mlflow.pyfunc
from sentence_transformers import SentenceTransformer
import redis
from redis import Redis

from utils.logger import get_logger

logger = get_logger(__name__)

class FeatureType(Enum):
    """Types of features in the feature store."""
    EMBEDDING = "embedding"
    SCHEMA = "schema"
    QUERY_PATTERN = "query_pattern"
    METADATA = "metadata"
    STATISTICS = "statistics"

class EmbeddingModel(Enum):
    """Supported embedding models."""
    SENTENCE_TRANSFORMERS = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_ADA = "text-embedding-ada-002"
    CUSTOM = "custom"

@dataclass
class FeatureMetadata:
    """Metadata for features stored in the feature store."""
    feature_id: str
    feature_type: FeatureType
    model_name: str
    model_version: str
    collection_name: str
    created_at: datetime
    updated_at: datetime
    schema_version: str
    embedding_dimension: Optional[int] = None
    statistics: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None
    lineage: Optional[Dict[str, Any]] = None

@dataclass
class EmbeddingRecord:
    """Record for storing embeddings with metadata."""
    document_id: str
    collection_name: str
    text_content: str
    embedding: np.ndarray
    model_name: str
    model_version: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

class FeatureStore:
    """
    Production-grade feature store for managing embeddings and features.
    
    Features:
    - Versioned embeddings with model tracking
    - Schema discovery and caching
    - Query pattern analysis
    - Redis caching for fast retrieval
    - MLflow integration for model versioning
    - Lineage tracking for data provenance
    """
    
    def __init__(self, 
                 mongodb_uri: str,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 mlflow_tracking_uri: str = None):
        """
        Initialize the feature store.
        
        Args:
            mongodb_uri: MongoDB connection string
            redis_host: Redis host for caching
            redis_port: Redis port
            redis_db: Redis database number
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.mongodb_uri = mongodb_uri
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        
        # Initialize connections
        self._init_mongodb()
        self._init_redis()
        self._init_mlflow(mlflow_tracking_uri)
        
        # Initialize embedding models
        self._init_embedding_models()
        
        logger.info("Feature store initialized successfully")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection and collections."""
        try:
            self.mongo_client = MongoClient(self.mongodb_uri)
            self.db = self.mongo_client.get_default_database()
            
            # Create collections with proper indexes
            self.embeddings_collection = self.db.embeddings
            self.features_collection = self.db.features
            self.schemas_collection = self.db.schemas
            self.query_patterns_collection = self.db.query_patterns
            
            # Create indexes for performance
            self._create_indexes()
            
            logger.info("MongoDB connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise
    
    def _init_redis(self):
        """Initialize Redis connection for caching."""
        try:
            self.redis_client = Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_mlflow(self, mlflow_tracking_uri: str):
        """Initialize MLflow for model tracking."""
        try:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            # Set experiment
            self.experiment_name = "mongodb-query-translator-features"
            mlflow.set_experiment(self.experiment_name)
            
            logger.info("MLflow initialized")
            
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
    
    def _init_embedding_models(self):
        """Initialize embedding models."""
        self.embedding_models = {}
        
        try:
            # Load sentence transformers model
            model_name = EmbeddingModel.SENTENCE_TRANSFORMERS.value
            self.embedding_models[model_name] = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        try:
            # Embeddings collection indexes
            self.embeddings_collection.create_index([
                ("collection_name", ASCENDING),
                ("model_name", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            self.embeddings_collection.create_index([
                ("document_id", ASCENDING),
                ("collection_name", ASCENDING)
            ])
            
            # Features collection indexes
            self.features_collection.create_index([
                ("feature_type", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            # Schemas collection indexes
            self.schemas_collection.create_index([
                ("collection_name", ASCENDING),
                ("schema_version", ASCENDING)
            ])
            
            logger.info("Database indexes created")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def generate_embedding(self, 
                          text: str, 
                          model_name: str = None,
                          cache: bool = True) -> np.ndarray:
        """
        Generate embedding for text using specified model.
        
        Args:
            text: Input text to embed
            model_name: Name of the embedding model to use
            cache: Whether to use Redis caching
            
        Returns:
            numpy array representing the embedding
        """
        if model_name is None:
            model_name = EmbeddingModel.SENTENCE_TRANSFORMERS.value
        
        # Check cache first
        if cache and self.redis_client:
            cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}:{model_name}"
            cached_embedding = self.redis_client.get(cache_key)
            if cached_embedding:
                return np.frombuffer(cached_embedding.encode('latin1'), dtype=np.float32)
        
        try:
            # Generate embedding
            if model_name in self.embedding_models:
                model = self.embedding_models[model_name]
                embedding = model.encode(text)
            else:
                raise ValueError(f"Model {model_name} not available")
            
            # Cache the result
            if cache and self.redis_client:
                cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}:{model_name}"
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour TTL
                    embedding.tobytes().decode('latin1')
                )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def store_embedding(self, 
                       document_id: str,
                       collection_name: str,
                       text_content: str,
                       model_name: str = None,
                       metadata: Dict[str, Any] = None) -> str:
        """
        Store embedding in the feature store.
        
        Args:
            document_id: Unique identifier for the document
            collection_name: Name of the MongoDB collection
            text_content: Text content to embed
            model_name: Name of the embedding model
            metadata: Additional metadata
            
        Returns:
            Feature ID of the stored embedding
        """
        if model_name is None:
            model_name = EmbeddingModel.SENTENCE_TRANSFORMERS.value
        
        try:
            # Generate embedding
            embedding = self.generate_embedding(text_content, model_name)
            
            # Create embedding record
            embedding_record = EmbeddingRecord(
                document_id=document_id,
                collection_name=collection_name,
                text_content=text_content,
                embedding=embedding,
                model_name=model_name,
                model_version="1.0",  # TODO: Get from MLflow
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {}
            )
            
            # Store in MongoDB
            embedding_doc = {
                "document_id": embedding_record.document_id,
                "collection_name": embedding_record.collection_name,
                "text_content": embedding_record.text_content,
                "embedding": embedding_record.embedding.tolist(),
                "model_name": embedding_record.model_name,
                "model_version": embedding_record.model_version,
                "created_at": embedding_record.created_at,
                "metadata": embedding_record.metadata
            }
            
            result = self.embeddings_collection.insert_one(embedding_doc)
            feature_id = str(result.inserted_id)
            
            # Store feature metadata
            feature_metadata = FeatureMetadata(
                feature_id=feature_id,
                feature_type=FeatureType.EMBEDDING,
                model_name=model_name,
                model_version="1.0",
                collection_name=collection_name,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                schema_version="1.0",
                embedding_dimension=len(embedding),
                statistics={
                    "text_length": len(text_content),
                    "word_count": len(text_content.split())
                },
                tags={"type": "document_embedding"},
                lineage={
                    "source": "mongodb_document",
                    "transformation": "sentence_transformer",
                    "model": model_name
                }
            )
            
            self.features_collection.insert_one(asdict(feature_metadata))
            
            logger.info(f"Stored embedding for document {document_id}")
            return feature_id
            
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            raise
    
    def get_embedding(self, 
                     document_id: str,
                     collection_name: str,
                     model_name: str = None) -> Optional[EmbeddingRecord]:
        """
        Retrieve embedding from the feature store.
        
        Args:
            document_id: Document identifier
            collection_name: Collection name
            model_name: Model name filter
            
        Returns:
            EmbeddingRecord if found, None otherwise
        """
        try:
            query = {
                "document_id": document_id,
                "collection_name": collection_name
            }
            
            if model_name:
                query["model_name"] = model_name
            
            doc = self.embeddings_collection.find_one(query)
            
            if doc:
                return EmbeddingRecord(
                    document_id=doc["document_id"],
                    collection_name=doc["collection_name"],
                    text_content=doc["text_content"],
                    embedding=np.array(doc["embedding"]),
                    model_name=doc["model_name"],
                    model_version=doc["model_version"],
                    created_at=doc["created_at"],
                    metadata=doc.get("metadata", {})
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding: {e}")
            return None
    
    def semantic_search(self, 
                       query: str,
                       collection_name: str,
                       model_name: str = None,
                       top_k: int = 10,
                       similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            collection_name: Collection to search in
            model_name: Model name for consistency
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query, model_name)
            
            # Get all embeddings for the collection
            embeddings_cursor = self.embeddings_collection.find({
                "collection_name": collection_name
            })
            
            results = []
            for doc in embeddings_cursor:
                doc_embedding = np.array(doc["embedding"])
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                if similarity >= similarity_threshold:
                    results.append({
                        "document_id": doc["document_id"],
                        "text_content": doc["text_content"],
                        "similarity_score": float(similarity),
                        "metadata": doc.get("metadata", {}),
                        "created_at": doc["created_at"]
                    })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def discover_schema(self, collection_name: str) -> Dict[str, Any]:
        """
        Discover and cache schema for a MongoDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Schema information
        """
        try:
            # Check if schema is already cached
            cached_schema = self.schemas_collection.find_one({
                "collection_name": collection_name
            }, sort=[("created_at", DESCENDING)])
            
            if cached_schema:
                logger.info(f"Using cached schema for {collection_name}")
                return cached_schema["schema"]
            
            # Discover schema
            collection = self.db[collection_name]
            sample_docs = list(collection.find().limit(100))
            
            if not sample_docs:
                return {"error": "No documents found"}
            
            # Analyze fields
            field_analysis = {}
            for doc in sample_docs:
                for key, value in doc.items():
                    if key not in field_analysis:
                        field_analysis[key] = {
                            "type": type(value).__name__,
                            "count": 0,
                            "sample_values": []
                        }
                    
                    field_analysis[key]["count"] += 1
                    if len(field_analysis[key]["sample_values"]) < 5:
                        field_analysis[key]["sample_values"].append(str(value)[:100])
            
            # Calculate statistics
            schema = {
                "collection_name": collection_name,
                "total_documents": collection.count_documents({}),
                "sampled_documents": len(sample_docs),
                "fields": field_analysis,
                "indexes": list(collection.list_indexes()),
                "discovered_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Cache the schema
            self.schemas_collection.insert_one({
                "collection_name": collection_name,
                "schema": schema,
                "schema_version": "1.0",
                "created_at": datetime.now(timezone.utc)
            })
            
            logger.info(f"Schema discovered and cached for {collection_name}")
            return schema
            
        except Exception as e:
            logger.error(f"Schema discovery failed: {e}")
            return {"error": str(e)}
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about the feature store."""
        try:
            stats = {
                "embeddings_count": self.embeddings_collection.count_documents({}),
                "features_count": self.features_collection.count_documents({}),
                "schemas_count": self.schemas_collection.count_documents({}),
                "collections_with_embeddings": list(
                    self.embeddings_collection.distinct("collection_name")
                ),
                "models_used": list(
                    self.embeddings_collection.distinct("model_name")
                ),
                "redis_connected": self.redis_client is not None,
                "mlflow_connected": mlflow.get_tracking_uri() is not None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close connections."""
        try:
            if hasattr(self, 'mongo_client'):
                self.mongo_client.close()
            if hasattr(self, 'redis_client') and self.redis_client:
                self.redis_client.close()
            logger.info("Feature store connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
