"""
Simple validation engine for MongoDB queries.
"""

import logging
from typing import Dict, Any
from config.database import get_database
from services.query_generator import QueryGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

class ValidationEngine:
    """Simple validation engine for MongoDB queries."""
    
    def __init__(self):
        self.db = get_database()
        self.query_generator = QueryGenerator()
    
    def validate_and_execute(self, user_query: str, execute_query: bool = True) -> Dict[str, Any]:
        """
        Validate and optionally execute a query.
        
        Args:
            user_query: Natural language query
            execute_query: Whether to execute the query
            
        Returns:
            Dict with results
        """
        try:
            # Generate query
            result = self.query_generator.generate_query(user_query)
            
            if not result["success"]:
                return result
            
            mql = result["generated_mql"]
            
            # Simple validation
            if not self._validate_query(mql):
                return {
                    "success": False,
                    "user_query": user_query,
                    "error": "Invalid query structure"
                }
            
            # Execute query if requested
            if execute_query:
                execution_result = self._execute_query(mql)
                result["execution_result"] = execution_result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return {
                "success": False,
                "user_query": user_query,
                "error": str(e)
            }
    
    def _validate_query(self, mql: Dict[str, Any]) -> bool:
        """Simple query validation."""
        required_fields = ["type", "collection"]
        
        # Check required fields
        for field in required_fields:
            if field not in mql:
                return False
        
        # Check query type
        if mql["type"] not in ["find", "aggregate"]:
            return False
        
        # Check collection exists
        collections = self.db.list_collection_names()
        if mql["collection"] not in collections and collections:  # Allow if no collections exist
            logger.warning(f"Collection {mql['collection']} not found")
        
        return True
    
    def _execute_query(self, mql: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MongoDB query."""
        try:
            collection = self.db[mql["collection"]]
            
            if mql["type"] == "find":
                filter_query = mql.get("filter", {})
                results = list(collection.find(filter_query).limit(100))
                return {
                    "success": True,
                    "results": results,
                    "count": len(results)
                }
            
            elif mql["type"] == "aggregate":
                pipeline = mql.get("pipeline", [])
                results = list(collection.aggregate(pipeline))
                return {
                    "success": True,
                    "results": results,
                    "count": len(results)
                }
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "success": False,
                "error": str(e)
            }