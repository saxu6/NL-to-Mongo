# API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required. In production, implement proper API key authentication.

## Endpoints

### 1. Root Endpoint
**GET** `/`

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "MongoDB Query Translator API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "generate_query": "/query/generate",
    "validate_query": "/query/validate",
    "execute_query": "/query/execute",
    "statistics": "/stats",
    "database_info": "/database/info",
    "schema_info": "/schema/info"
  },
  "timestamp": "2025-09-08T17:45:35.716069"
}
```

### 2. Health Check
**GET** `/health`

Returns system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-08T17:42:52.179821",
  "database_connected": true,
  "components_initialized": {
    "query_generator": true,
    "validation_engine": true,
    "openai_client": true
  },
  "validation_stats": {
    "total_validations": 0,
    "successful_validations": 0,
    "failed_validations": 0,
    "corrections_attempted": 0,
    "successful_corrections": 0,
    "failed_corrections": 0,
    "success_rate": 0.0,
    "correction_success_rate": 0.0
  }
}
```

### 3. Generate Query
**POST** `/query/generate`

Converts natural language to MongoDB query.

**Request Body:**
```json
{
  "query": "Find all active users",
  "max_retries": 3,
  "include_alternatives": true,
  "execute_query": false
}
```

**Response:**
```json
{
  "success": true,
  "user_query": "Find all active users",
  "generated_mql": {
    "type": "find",
    "collection": "users",
    "filter": {"status": "active"},
    "projection": null,
    "sort": null,
    "limit": null
  },
  "query_type": "find",
  "explanation": "This query finds all documents in the 'users' collection where the status field equals 'active'.",
  "confidence": 0.95,
  "suggested_indexes": ["status"],
  "validation_result": {
    "valid": true,
    "errors": [],
    "warnings": []
  },
  "execution_result": null,
  "alternative_queries": [
    {
      "description": "Alternative with projection",
      "mql": {
        "type": "find",
        "collection": "users",
        "filter": {"status": "active"},
        "projection": {"name": 1, "email": 1, "_id": 0}
      }
    }
  ],
  "timestamp": "2025-09-08T17:43:04.140505"
}
```

### 4. Validate Query
**POST** `/query/validate`

Validates a MongoDB query.

**Request Body:**
```json
{
  "mql": {
    "type": "find",
    "collection": "users",
    "filter": {"status": "active"}
  }
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "suggestions": ["Consider adding index on 'status' field"],
  "performance_analysis": {
    "estimated_documents_examined": 1000,
    "estimated_documents_returned": 100,
    "index_usage": "none"
  }
}
```

### 5. Execute Query
**POST** `/query/execute`

Executes a MongoDB query and returns results.

**Request Body:**
```json
{
  "mql": {
    "type": "find",
    "collection": "users",
    "filter": {"status": "active"},
    "limit": 10
  }
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "name": "John Doe",
      "email": "john@example.com",
      "status": "active"
    }
  ],
  "count": 1,
  "execution_time_ms": 45,
  "query_plan": {
    "stage": "COLLSCAN",
    "filter": {"status": {"$eq": "active"}}
  }
}
```

### 6. Statistics
**GET** `/stats`

Returns system statistics.

**Response:**
```json
{
  "total_queries": 150,
  "successful_queries": 142,
  "failed_queries": 8,
  "success_rate": 0.947,
  "average_response_time_ms": 1250,
  "most_common_queries": [
    {
      "query": "Find all active users",
      "count": 25
    }
  ],
  "validation_stats": {
    "total_validations": 150,
    "successful_validations": 142,
    "failed_validations": 8,
    "corrections_attempted": 8,
    "successful_corrections": 6,
    "failed_corrections": 2
  }
}
```

### 7. Database Info
**GET** `/database/info`

Returns database information.

**Response:**
```json
{
  "database_name": "production_db",
  "collections": ["users", "orders", "products"],
  "collection_stats": {
    "users": {
      "count": 10000,
      "size_bytes": 5242880,
      "indexes": 3
    },
    "orders": {
      "count": 50000,
      "size_bytes": 10485760,
      "indexes": 5
    }
  },
  "timestamp": "2025-09-08T17:42:59.456013"
}
```

### 8. Schema Info
**GET** `/schema/info`

Returns schema information.

**Response:**
```json
{
  "collections": {
    "users": {
      "fields": {
        "name": {"type": "string", "description": "User's full name"},
        "email": {"type": "string", "description": "User's email address"},
        "status": {"type": "string", "description": "User account status"},
        "created_at": {"type": "date", "description": "Account creation timestamp"}
      },
      "indexes": ["email", "status", "created_at"],
      "sample_documents": 1
    }
  },
  "timestamp": "2025-09-08T17:42:59.456013"
}
```

## Error Responses

All endpoints return errors in the following format:

```json
{
  "success": false,
  "error": "Error message description",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-09-08T17:43:04.140505"
}
```

## Rate Limiting

Currently no rate limiting implemented. In production, implement rate limiting based on API keys.

## CORS

CORS is enabled for all origins. In production, restrict to specific domains.
