"""
Simplified Embedding Service for MongoDB Query Translator Demo.
Provides basic embedding capabilities for demonstration purposes.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.logger import get_logger
from config.database import get_database

logger = get_logger(__name__)

class EmbeddingService:
    """
    Simplified embedding service for demo purposes.
    Uses basic text processing and similarity calculations.
    """
    
    def __init__(self):
        self.db = get_database()
        self.model_name = "demo-embedding-model"
        self.embeddings_cache = {}
        self.query_embeddings = {}
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a simple embedding for demonstration purposes.
        Uses basic text features as embedding dimensions.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array representing the embedding
        """
        try:
            if text in self.embeddings_cache:
                return self.embeddings_cache[text]
            
            # Simple embedding based on text features
            text_lower = text.lower()
            
            # Create feature vector
            features = []
            
            # Word count
            word_count = len(text.split())
            features.append(word_count / 20.0)  # Normalize
            
            # Character count
            char_count = len(text)
            features.append(char_count / 100.0)  # Normalize
            
            # Common query words
            query_words = ['find', 'get', 'show', 'count', 'sort', 'filter', 'group', 'aggregate']
            word_features = [1.0 if word in text_lower else 0.0 for word in query_words]
            features.extend(word_features)
            
            # MongoDB operators
            mongo_ops = ['$gt', '$lt', '$eq', '$in', '$and', '$or', '$match', '$group', '$sort']
            op_features = [1.0 if op in text_lower else 0.0 for op in mongo_ops]
            features.extend(op_features)
            
            # Collection names (common ones)
            collections = ['users', 'products', 'orders', 'transactions', 'documents']
            coll_features = [1.0 if coll in text_lower else 0.0 for coll in collections]
            features.extend(coll_features)
            
            # Field names (common ones)
            fields = ['name', 'email', 'date', 'price', 'status', 'category', 'id', 'created']
            field_features = [1.0 if field in text_lower else 0.0 for field in fields]
            features.extend(field_features)
            
            # Pad or truncate to fixed size
            target_size = 50
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            embedding = np.array(features, dtype=np.float32)
            
            # Cache the embedding
            self.embeddings_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(50, dtype=np.float32)
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about the embedding system."""
        return {
            "model_name": self.model_name,
            "cache_size": len(self.embeddings_cache),
            "index_available": False,
            "index_size": 0,
            "embedding_dimension": 50,
            "memory_usage": {
                "rss_mb": 0.0,
                "vms_mb": 0.0,
                "percent": 0.0
            }
        }
    
    def semantic_search(self, query: str, collection_name: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform basic semantic search for demo purposes.
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Get sample documents from collection
            collection = self.db[collection_name]
            documents = list(collection.find().limit(100))
            
            if not documents:
                return {
                    "success": True,
                    "query": query,
                    "collection": collection_name,
                    "results": [],
                    "total_results": 0,
                    "search_method": "demo_semantic_search"
                }
            
            # Calculate similarities (simplified)
            results = []
            for i, doc in enumerate(documents[:top_k]):
                # Create a simple document representation
                doc_text = str(doc)
                doc_embedding = self.generate_embedding(doc_text)
                
                # Calculate cosine similarity
                similarity = float(np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8
                ))
                
                results.append({
                    "rank": i + 1,
                    "document": doc,
                    "similarity_score": similarity,
                    "document_id": str(doc.get('_id', ''))
                })
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return {
                "success": True,
                "query": query,
                "collection": collection_name,
                "results": results,
                "total_results": len(results),
                "search_method": "demo_semantic_search"
            }
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {"error": str(e)}
    
    def compare_search_methods(self, query: str, collection_name: str) -> Dict[str, Any]:
        """
        Compare different search methods for demo purposes.
        """
        try:
            # Perform semantic search
            semantic_results = self.semantic_search(query, collection_name, top_k=5)
            
            # Simulate traditional search
            collection = self.db[collection_name]
            traditional_results = list(collection.find().limit(5))
            
            # Simulate fuzzy search
            fuzzy_results = list(collection.find().limit(5))
            
            return {
                "success": True,
                "query": query,
                "collection": collection_name,
                "comparison": {
                    "semantic_search": {
                        "method": "Demo Vector Embeddings",
                        "results_count": len(semantic_results.get("results", [])),
                        "avg_similarity": np.mean([r["similarity_score"] for r in semantic_results.get("results", [])]) if semantic_results.get("results") else 0,
                        "pros": ["Semantic understanding", "Handles synonyms", "Context-aware"],
                        "cons": ["Requires preprocessing", "Computational overhead"]
                    },
                    "traditional_search": {
                        "method": "MongoDB Native",
                        "results_count": len(traditional_results),
                        "pros": ["Fast", "Native MongoDB", "Exact matching"],
                        "cons": ["Limited semantic understanding", "Requires exact matches"]
                    },
                    "fuzzy_search": {
                        "method": "Pattern Matching",
                        "results_count": len(fuzzy_results),
                        "pros": ["Flexible patterns", "No preprocessing", "Fast"],
                        "cons": ["Limited semantic understanding", "Pattern dependent"]
                    }
                },
                "recommendation": "Semantic search recommended for better relevance and context understanding"
            }
            
        except Exception as e:
            logger.error(f"Error comparing search methods: {e}")
            return {"error": str(e)}
    
    def build_collection_embeddings(self, collection_name: str) -> Dict[str, Any]:
        """
        Build embeddings for a collection (demo version).
        """
        try:
            collection = self.db[collection_name]
            documents = list(collection.find().limit(100))
            
            if not documents:
                return {"error": "No documents found in collection"}
            
            # Generate embeddings for sample documents
            embeddings_count = 0
            for doc in documents[:10]:  # Limit for demo
                doc_text = str(doc)
                self.generate_embedding(doc_text)
                embeddings_count += 1
            
            return {
                "success": True,
                "collection_name": collection_name,
                "documents_processed": len(documents),
                "embeddings_created": embeddings_count,
                "dimension": 50,
                "index_type": "demo_embeddings"
            }
            
        except Exception as e:
            logger.error(f"Error building collection embeddings: {e}")
            return {"error": str(e)}
