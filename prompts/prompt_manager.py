"""
Simple prompt manager for LLM queries.
"""

from utils.logger import get_logger

logger = get_logger(__name__)

class PromptManager:
    """Simple prompt manager for generating LLM prompts."""
    
    def create_main_prompt(self, user_query: str, schema_info: str = "") -> str:
        """Create main prompt for query generation."""
        return f"""Convert this natural language query to MongoDB query JSON format.

Query: {user_query}

{schema_info if schema_info else "No schema information available."}

Return ONLY valid JSON in this exact format:
{{
    "type": "find",
    "collection": "collection_name",
    "filter": {{"field": "value"}},
    "explanation": "brief explanation"
}}

For aggregation queries use:
{{
    "type": "aggregate", 
    "collection": "collection_name",
    "pipeline": [{{"$match": {{"field": "value"}}}}],
    "explanation": "brief explanation"
}}

Return ONLY the JSON, no other text."""
