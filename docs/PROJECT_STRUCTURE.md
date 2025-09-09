# Project Structure

## 📁 Complete File Organization

```
nl_to_mongo_new/
├── 📁 api/                          # FastAPI Backend Server
│   ├── __init__.py
│   └── main.py                      # Main API server with all endpoints
│
├── 📁 atlas/                        # MongoDB Atlas Integration
│   ├── __init__.py
│   └── index_manager.py             # Atlas Search index management
│
├── 📁 config/                       # Configuration Management
│   ├── __init__.py
│   └── database.py                  # Database connection and management
│
├── 📁 docs/                         # Documentation
│   ├── API_REFERENCE.md             # Complete API documentation
│   ├── DEPLOYMENT.md                # Production deployment guide
│   ├── PROJECT_STRUCTURE.md         # This file
│   ├── phase1_step1_environment_setup.md
│   ├── phase1_step2_schema_discovery.md
│   ├── phase1_step3_dual_embedding_strategy.md
│   ├── phase1_step4_atlas_index_creation.md
│   ├── phase2_step1_prompt_system.md
│   ├── phase2_step2_query_generation_service.md
│   ├── phase2_step3_validation_engine.md
│   ├── phase3_step1_frontend_api.md
│   ├── phase3_step3_final_system_summary.md
│   ├── prompt_system_test_summary.json
│   └── validation_engine_test_summary.json
│
├── 📁 embeddings/                   # Vector Embeddings & Search
│   ├── __init__.py
│   ├── embedding_manager.py         # Embedding generation and management
│   └── vector_search.py             # Vector search operations
│
├── 📁 prompts/                      # LLM Prompt Templates
│   ├── __init__.py
│   ├── prompt_manager.py            # Prompt template management
│   └── 📁 templates/
│       ├── correction_prompt.j2     # Query correction prompts
│       ├── main_prompt.j2           # Main query generation prompts
│       └── validation_prompt.j2     # Query validation prompts
│
├── 📁 schema/                       # Schema Discovery & Analysis
│   ├── __init__.py
│   ├── schema_analyzer.py           # Schema analysis and insights
│   └── schema_discovery.py          # Dynamic schema discovery
│
├── 📁 scripts/                      # Utility Scripts
│   ├── run_tests.py                 # Test runner script
│   └── start_server.py              # Server startup script
│
├── 📁 services/                     # Core Business Logic
│   ├── __init__.py
│   ├── query_generator.py           # Natural language to MQL conversion
│   └── validation_engine.py         # Query validation and self-correction
│
├── 📁 tests/                        # Test Suite
│   ├── test_atlas_indexes.py        # Atlas index tests
│   ├── test_connection.py           # Database connection tests
│   ├── test_embeddings.py           # Embedding system tests
│   ├── test_prompt_system.py        # Prompt system tests
│   ├── test_query_generator.py      # Query generation tests
│   ├── test_schema_analysis.py      # Schema analysis tests
│   ├── test_schema_discovery.py     # Schema discovery tests
│   ├── test_validation_engine.py    # Validation engine tests
│   └── test_vector_search.py        # Vector search tests
│
├── 📁 utils/                        # Utility Functions
│   ├── __init__.py
│   └── logger.py                    # Logging configuration
│
├── 📁 visualization/                # Data Visualization
│   ├── __init__.py
│   └── data_visualizer.py           # Intelligent chart selection
│
├── 📄 .env                          # Environment variables (local)
├── 📄 .env.backup                   # Backup of original env file
├── 📄 env.example                   # Environment template
├── 📄 README.md                     # Main project documentation
└── 📄 requirements.txt              # Python dependencies
```

## 🏗️ Architecture Overview

### **Core Components**

1. **API Layer** (`api/`)
   - FastAPI server with REST endpoints
   - Request/response handling
   - Error management

2. **Business Logic** (`services/`)
   - Query generation from natural language
   - Validation and self-correction
   - Core AI/ML processing

3. **Data Layer** (`config/`, `schema/`)
   - Database connections
   - Schema discovery and analysis
   - Data structure management

4. **AI/ML Layer** (`embeddings/`, `prompts/`)
   - Vector embeddings generation
   - LLM prompt management
   - Semantic search capabilities

5. **Infrastructure** (`atlas/`, `utils/`)
   - MongoDB Atlas integration
   - Logging and utilities
   - Index management

6. **Presentation** (`visualization/`)
   - Data visualization
   - Chart generation
   - Result formatting

### **Supporting Components**

- **Tests** (`tests/`): Comprehensive test suite
- **Scripts** (`scripts/`): Utility and deployment scripts
- **Documentation** (`docs/`): Complete project documentation

## 🔄 Data Flow

```
User Query → API → Query Generator → Validation Engine → MongoDB → Results → Visualizer → Response
     ↓           ↓         ↓              ↓              ↓         ↓          ↓
   Natural    FastAPI   LLM +        Self-Correct   Atlas DB   Charts/    JSON
   Language   Server    Embeddings   Validation              Tables     Response
```

## 📊 File Statistics

- **Total Files**: 45
- **Python Files**: 25
- **Documentation**: 15
- **Templates**: 3
- **Configuration**: 2

## 🎯 Key Features by Directory

### **api/**
- RESTful API endpoints
- Request validation
- Response formatting
- Error handling

### **services/**
- Natural language processing
- Query generation
- Validation logic
- Self-correction mechanisms

### **embeddings/**
- Vector generation
- Semantic search
- Similarity matching
- Embedding storage

### **schema/**
- Dynamic schema discovery
- Field analysis
- Data type detection
- Business context mapping

### **prompts/**
- LLM prompt templates
- Few-shot learning examples
- Context management
- Template rendering

### **visualization/**
- Chart selection logic
- Data formatting
- Visualization generation
- Result presentation

## 🚀 Quick Start Commands

```bash
# Start the server
python scripts/start_server.py

# Run all tests
python scripts/run_tests.py

# Test specific component
python tests/test_connection.py

# View API documentation
# Open http://localhost:8000/docs
```

## 📝 Development Workflow

1. **Development**: Work in respective module directories
2. **Testing**: Add tests in `tests/` directory
3. **Documentation**: Update docs in `docs/` directory
4. **Deployment**: Use scripts in `scripts/` directory

## 🔧 Maintenance

- **Logs**: Check `logs/` directory (created at runtime)
- **Configuration**: Update `.env` file
- **Dependencies**: Update `requirements.txt`
- **Documentation**: Keep `docs/` current

This structure provides a clean, maintainable, and scalable foundation for the MongoDB Query Translator system.
