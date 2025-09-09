# MongoDB Query Translator with Ollama

A production-ready AI-powered system that converts natural language queries into optimized MongoDB queries using local Ollama models - no external API keys required!

## 🚀 Features

- **Natural Language Processing**: Convert English to MongoDB queries
- **Local AI**: Uses Ollama with gpt-oss:20b model (no API costs!)
- **Intelligent Validation**: Self-correcting query validation system
- **Production Ready**: Complete REST API with Docker support
- **Self-Contained**: No external dependencies or API keys
- **Privacy-First**: All data stays local

## 📁 Project Structure

```
nl_to_mongo_new/
├── api/                    # FastAPI backend server
├── config/                 # Database and application configuration
├── docs/                   # Documentation
├── prompts/                # LLM prompt templates
├── schema/                 # Schema discovery and analysis
├── scripts/                # Utility scripts and deployment
├── services/               # Core business logic
├── tests/                  # Test suite
├── utils/                  # Utility functions
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Quick Start

### Docker (Recommended)
```bash
# Clone and start
git clone <repository-url>
cd nl_to_mongo_new
cp env.example .env
# Edit .env with your MongoDB URI

# Deploy with Ollama
./scripts/deploy.sh
```

### Manual Setup
```bash
# Start Ollama
docker-compose up -d ollama

# Pull the model
docker exec -it ollama ollama pull gpt-oss:20b

# Start the application
docker-compose up -d mongodb-query-translator
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama locally
ollama serve

# Pull the model
ollama pull gpt-oss:20b

# Start server
python scripts/start_server.py
```

## 🎯 Usage

### API Endpoints

The server runs on `http://localhost:8000` with these endpoints:

- **GET** `/` - API information
- **GET** `/health` - System health check
- **POST** `/query/generate` - Generate MongoDB queries
- **POST** `/query/execute` - Execute queries
- **GET** `/database/info` - Database information

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Generate a query
curl -X POST http://localhost:8000/query/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Find all active users", "execute_query": false}'
```

## 🧪 Testing

Run the test suite:

```bash
# Run basic tests
python tests/test_basic.py

# Run all tests
python scripts/run_tests.py
```

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Natural       │    │   FastAPI       │    │   MongoDB       │
│   Language      │───▶│   Backend       │───▶│   Database      │
│   Input         │    │   (Port 8000)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Ollama        │
                       │   gpt-oss:20b   │
                       │   (Port 11434)  │
                       └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

- `MONGODB_URI`: MongoDB Atlas connection string
- `OLLAMA_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (default: gpt-oss:20b)
- `ATLAS_DATABASE_NAME`: Database name

### Dependencies

- **FastAPI**: Web framework
- **PyMongo**: MongoDB driver
- **Ollama**: Local LLM integration
- **Requests**: HTTP client

## 📈 Performance

- **Query Generation**: < 3 seconds (local inference)
- **Validation**: < 1 second
- **API Response**: < 4 seconds
- **No API Costs**: Completely free to run

## 🏆 Benefits of Using Ollama

1. **No API Costs**: Completely free to run
2. **Privacy**: All data stays local
3. **Offline**: Works without internet
4. **Customizable**: Use any model you want
5. **Fast**: Local inference is very fast
6. **Self-contained**: No external dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Check the documentation in `docs/`
- Run the test suite to verify functionality
- Review the API endpoints for usage examples

## 🎉 Acknowledgments

Built with local AI using Ollama and gpt-oss:20b model for complete privacy and cost-effectiveness.
