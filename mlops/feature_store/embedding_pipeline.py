"""
Automated Embedding Pipeline for MongoDB Query Translator.
Processes documents and generates embeddings in batch with monitoring.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError
import mlflow
import mlflow.pyfunc

from feature_store import FeatureStore, EmbeddingModel
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ProcessingStats:
    """Statistics for embedding processing."""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    skipped_documents: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100

class EmbeddingPipeline:
    """
    Automated pipeline for processing MongoDB documents and generating embeddings.
    
    Features:
    - Batch processing with configurable batch sizes
    - Parallel processing for improved performance
    - Progress tracking and monitoring
    - Error handling and retry logic
    - MLflow experiment tracking
    - Resume capability for interrupted runs
    """
    
    def __init__(self, 
                 feature_store: FeatureStore,
                 batch_size: int = 100,
                 max_workers: int = 4,
                 retry_attempts: int = 3):
        """
        Initialize the embedding pipeline.
        
        Args:
            feature_store: Feature store instance
            batch_size: Number of documents to process in each batch
            max_workers: Maximum number of parallel workers
            retry_attempts: Number of retry attempts for failed documents
        """
        self.feature_store = feature_store
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        
        # Initialize MLflow experiment
        self.experiment_name = "embedding-pipeline"
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"Embedding pipeline initialized with batch_size={batch_size}, max_workers={max_workers}")
    
    def process_collection(self, 
                          collection_name: str,
                          model_name: str = None,
                          filter_query: Dict[str, Any] = None,
                          text_fields: List[str] = None,
                          resume_from: Optional[str] = None) -> ProcessingStats:
        """
        Process all documents in a collection to generate embeddings.
        
        Args:
            collection_name: Name of the MongoDB collection
            model_name: Embedding model to use
            filter_query: MongoDB query to filter documents
            text_fields: List of fields to concatenate for embedding
            resume_from: Document ID to resume from (for interrupted runs)
            
        Returns:
            ProcessingStats with processing results
        """
        if model_name is None:
            model_name = EmbeddingModel.SENTENCE_TRANSFORMERS.value
        
        if text_fields is None:
            text_fields = ["text", "content", "description", "title"]
        
        stats = ProcessingStats()
        stats.start_time = datetime.now(timezone.utc)
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=f"embedding_pipeline_{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params({
                    "collection_name": collection_name,
                    "model_name": model_name,
                    "batch_size": self.batch_size,
                    "max_workers": self.max_workers,
                    "text_fields": text_fields
                })
                
                # Get collection
                collection = self.feature_store.db[collection_name]
                
                # Build query
                query = filter_query or {}
                if resume_from:
                    query["_id"] = {"$gt": resume_from}
                
                # Get total count
                stats.total_documents = collection.count_documents(query)
                logger.info(f"Processing {stats.total_documents} documents from {collection_name}")
                
                # Process in batches
                cursor = collection.find(query).batch_size(self.batch_size)
                
                for batch in self._batch_iterator(cursor, self.batch_size):
                    batch_stats = self._process_batch(
                        batch, collection_name, model_name, text_fields
                    )
                    
                    # Update overall stats
                    stats.processed_documents += batch_stats["processed"]
                    stats.failed_documents += batch_stats["failed"]
                    stats.skipped_documents += batch_stats["skipped"]
                    
                    # Log progress
                    progress = (stats.processed_documents + stats.failed_documents + stats.skipped_documents) / stats.total_documents * 100
                    logger.info(f"Progress: {progress:.1f}% ({stats.processed_documents + stats.failed_documents + stats.skipped_documents}/{stats.total_documents})")
                
                stats.end_time = datetime.now(timezone.utc)
                
                # Log final metrics
                mlflow.log_metrics({
                    "total_documents": stats.total_documents,
                    "processed_documents": stats.processed_documents,
                    "failed_documents": stats.failed_documents,
                    "skipped_documents": stats.skipped_documents,
                    "success_rate": stats.success_rate,
                    "duration_seconds": stats.duration
                })
                
                logger.info(f"Processing completed. Success rate: {stats.success_rate:.1f}%")
                return stats
                
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            stats.end_time = datetime.now(timezone.utc)
            raise
    
    def _batch_iterator(self, cursor: Iterator, batch_size: int) -> Iterator[List[Dict]]:
        """Convert cursor to batch iterator."""
        batch = []
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def _process_batch(self, 
                      batch: List[Dict[str, Any]], 
                      collection_name: str,
                      model_name: str,
                      text_fields: List[str]) -> Dict[str, int]:
        """
        Process a batch of documents.
        
        Args:
            batch: List of documents to process
            collection_name: Collection name
            model_name: Model name
            text_fields: Fields to extract text from
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {"processed": 0, "failed": 0, "skipped": 0}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all documents for processing
            future_to_doc = {
                executor.submit(
                    self._process_document, 
                    doc, collection_name, model_name, text_fields
                ): doc for doc in batch
            }
            
            # Collect results
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    result = future.result()
                    if result == "processed":
                        stats["processed"] += 1
                    elif result == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"Failed to process document {doc.get('_id')}: {e}")
                    stats["failed"] += 1
        
        return stats
    
    def _process_document(self, 
                         doc: Dict[str, Any],
                         collection_name: str,
                         model_name: str,
                         text_fields: List[str]) -> str:
        """
        Process a single document.
        
        Args:
            doc: Document to process
            collection_name: Collection name
            model_name: Model name
            text_fields: Fields to extract text from
            
        Returns:
            Processing result: "processed", "skipped", or "failed"
        """
        try:
            document_id = str(doc["_id"])
            
            # Check if embedding already exists
            existing_embedding = self.feature_store.get_embedding(
                document_id, collection_name, model_name
            )
            
            if existing_embedding:
                logger.debug(f"Embedding already exists for document {document_id}")
                return "skipped"
            
            # Extract text content
            text_content = self._extract_text_content(doc, text_fields)
            
            if not text_content or len(text_content.strip()) < 10:
                logger.debug(f"No meaningful text content found for document {document_id}")
                return "skipped"
            
            # Generate and store embedding
            self.feature_store.store_embedding(
                document_id=document_id,
                collection_name=collection_name,
                text_content=text_content,
                model_name=model_name,
                metadata={
                    "original_document": doc,
                    "text_fields_used": text_fields,
                    "text_length": len(text_content)
                }
            )
            
            return "processed"
            
        except Exception as e:
            logger.error(f"Failed to process document {doc.get('_id')}: {e}")
            return "failed"
    
    def _extract_text_content(self, 
                             doc: Dict[str, Any], 
                             text_fields: List[str]) -> str:
        """
        Extract text content from document fields.
        
        Args:
            doc: Document to extract text from
            text_fields: List of field names to extract
            
        Returns:
            Concatenated text content
        """
        text_parts = []
        
        for field in text_fields:
            if field in doc and doc[field]:
                value = doc[field]
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, (list, dict)):
                    text_parts.append(str(value))
        
        # If no specific fields found, try to extract from all string fields
        if not text_parts:
            for key, value in doc.items():
                if isinstance(value, str) and len(value) > 5 and key not in ["_id", "created_at", "updated_at"]:
                    text_parts.append(value)
        
        return " ".join(text_parts)
    
    def process_incremental(self, 
                           collection_name: str,
                           model_name: str = None,
                           since: Optional[datetime] = None) -> ProcessingStats:
        """
        Process only new or updated documents since a given timestamp.
        
        Args:
            collection_name: Collection name
            model_name: Model name
            since: Process documents modified since this timestamp
            
        Returns:
            ProcessingStats with results
        """
        if model_name is None:
            model_name = EmbeddingModel.SENTENCE_TRANSFORMERS.value
        
        # Build query for incremental processing
        filter_query = {}
        if since:
            filter_query["updated_at"] = {"$gte": since}
        
        logger.info(f"Processing incremental updates for {collection_name} since {since}")
        
        return self.process_collection(
            collection_name=collection_name,
            model_name=model_name,
            filter_query=filter_query
        )
    
    def validate_embeddings(self, 
                           collection_name: str,
                           sample_size: int = 100) -> Dict[str, Any]:
        """
        Validate embeddings for a collection.
        
        Args:
            collection_name: Collection name
            sample_size: Number of embeddings to validate
            
        Returns:
            Validation results
        """
        try:
            # Get sample of embeddings
            sample_embeddings = list(
                self.feature_store.embeddings_collection.find({
                    "collection_name": collection_name
                }).limit(sample_size)
            )
            
            if not sample_embeddings:
                return {"error": "No embeddings found"}
            
            validation_results = {
                "total_checked": len(sample_embeddings),
                "valid_embeddings": 0,
                "invalid_embeddings": 0,
                "issues": []
            }
            
            for embedding_doc in sample_embeddings:
                try:
                    # Check if embedding is valid
                    embedding = np.array(embedding_doc["embedding"])
                    
                    if len(embedding) == 0:
                        validation_results["invalid_embeddings"] += 1
                        validation_results["issues"].append({
                            "document_id": embedding_doc["document_id"],
                            "issue": "Empty embedding"
                        })
                    elif np.isnan(embedding).any() or np.isinf(embedding).any():
                        validation_results["invalid_embeddings"] += 1
                        validation_results["issues"].append({
                            "document_id": embedding_doc["document_id"],
                            "issue": "Invalid values (NaN or Inf)"
                        })
                    else:
                        validation_results["valid_embeddings"] += 1
                        
                except Exception as e:
                    validation_results["invalid_embeddings"] += 1
                    validation_results["issues"].append({
                        "document_id": embedding_doc["document_id"],
                        "issue": f"Processing error: {str(e)}"
                    })
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Embedding validation failed: {e}")
            return {"error": str(e)}
    
    def get_processing_report(self, collection_name: str) -> Dict[str, Any]:
        """Get a comprehensive report of processing status."""
        try:
            # Get collection stats
            collection = self.feature_store.db[collection_name]
            total_docs = collection.count_documents({})
            
            # Get embedding stats
            embedding_count = self.feature_store.embeddings_collection.count_documents({
                "collection_name": collection_name
            })
            
            # Get latest processing run
            latest_run = self.feature_store.features_collection.find_one({
                "collection_name": collection_name,
                "feature_type": "embedding"
            }, sort=[("created_at", -1)])
            
            report = {
                "collection_name": collection_name,
                "total_documents": total_docs,
                "embedded_documents": embedding_count,
                "coverage_percentage": (embedding_count / total_docs * 100) if total_docs > 0 else 0,
                "latest_processing": latest_run["created_at"] if latest_run else None,
                "feature_store_stats": self.feature_store.get_feature_statistics()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate processing report: {e}")
            return {"error": str(e)}
