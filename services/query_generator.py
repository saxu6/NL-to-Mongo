"""
Simple query generator using Ollama instead of OpenAI.
"""

import os
import json
import logging
from typing import Dict, Any
import requests

from config.database import get_database
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)

class QueryGenerator:
    """Simple service for converting natural language to MongoDB queries using Ollama."""
    
    def __init__(self):
        self.db = get_database()
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.model_name = os.getenv('OLLAMA_MODEL', 'gpt-oss:20b')
        self.prompt_manager = PromptManager()
    
    def generate_query(self, user_query: str) -> Dict[str, Any]:
        """
        Generate MongoDB query from natural language using Ollama.
        
        Args:
            user_query: Natural language query
            
        Returns:
            Dict with generated query and metadata
        """
        try:
            # Get schema info
            collections = self.db.list_collection_names()
            schema_info = f"Available collections: {', '.join(collections)}" if collections else "No collections found"
            
            # Create prompt
            prompt = self.prompt_manager.create_main_prompt(user_query, schema_info)
            
            # Call Ollama
            response = self._call_ollama(prompt)
            
            # Clean response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse response
            result = json.loads(cleaned_response)
            
            return {
                "success": True,
                "user_query": user_query,
                "generated_mql": result,
                "query_type": result.get("type"),
                "explanation": result.get("explanation"),
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            return {
                "success": False,
                "user_query": user_query,
                "error": str(e)
            }
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate response."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to Ollama: {e}")
