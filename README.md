# Enterprise Multi-Document RAG Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system designed for enterprise document analysis. Upload multiple documents, ask questions, and receive grounded answers with citations.

## Project Vision

This project will build a complete RAG system that enables:

- Multiple document ingestion (PDF, DOCX, TXT)
- Intelligent text chunking with metadata preservation
- Local embedding models for vector representation
- ChromaDB for efficient vector storage and retrieval
- LLM integration via OpenRouter for answer generation
- Source citations and document tracking
- Conversational interface with context awareness
- RAG quality evaluation and metrics

## Tech Stack

- **Backend**: FastAPI, Python
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (local)
- **LLM**: OpenRouter API
- **Frontend**: Next.js (upcoming)
- **Deployment**: Docker, Kubernetes (planned)

## Project Phases

### Phase 0: Project Foundation ✅

- Project structure setup
- Virtual environment configuration
- Initial dependencies
- Basic FastAPI application
- Environment variables setup

### Phase 1-3: Document Processing (Upcoming)

- PDF document ingestion
- Text chunking with metadata
- Embedding generation

### Phase 4-6: Vector Search & RAG (Upcoming)

- ChromaDB integration
- Vector storage and retrieval
- Basic RAG pipeline
- Citation tracking

### Phase 7-9: API & Multi-Document Support (Upcoming)

- FastAPI backend development
- Document management
- Multi-document queries

### Phase 10-17: Advanced Features & Deployment (Upcoming)

- Conversational RAG
- Retrieval optimization
- Reranking system
- LangGraph workflows
- RAG evaluation
- Docker deployment

## Getting Started

### Prerequisites

- Python 3.10+
- pip or conda
- OpenRouter API key (get free at https://openrouter.ai)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

2. **Create virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
cd backend
pip install -r requirements.txt
```

4. **Configure environment variables**

```bash
cp .env.example .env
# Edit .env with your OpenRouter API key
```

5. **Run the FastAPI server**

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
enterprise-rag-assistant/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI application
│   │   ├── api/              # API routes (upcoming)
│   │   ├── services/         # Business logic (upcoming)
│   │   ├── models/           # Data models (upcoming)
│   │   └── utils/            # Utility functions (upcoming)
│   │
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables (local)
│
├── frontend/                  # Next.js application (upcoming)
│
├── documents/                 # Sample documents for testing
│
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── docker-compose.yml         # Docker orchestration (upcoming)
```

## Environment Variables

| Variable              | Description                       | Example                        |
| --------------------- | --------------------------------- | ------------------------------ |
| `OPENROUTER_API_KEY`  | OpenRouter API key for LLM access | `sk-...`                       |
| `OPENROUTER_BASE_URL` | OpenRouter API endpoint           | `https://openrouter.ai/api/v1` |
| `LLM_MODEL`           | Model to use for generation       | `gpt-3.5-turbo`                |
| `LLM_MAX_TOKENS`      | Max tokens per response           | `2048`                         |
| `DEBUG`               | Enable debug mode                 | `True`                         |
| `LOG_LEVEL`           | Logging level                     | `INFO`                         |

## Key Concepts

### Retrieval-Augmented Generation (RAG)

A technique that combines document retrieval with large language models to provide grounded, accurate answers with source citations.

### Embeddings

Convert text into numerical vectors that capture semantic meaning, enabling similarity search.

### Vector Database

ChromaDB stores and retrieves vectors efficiently, supporting semantic search across documents.

### Chunking

Break documents into manageable pieces while preserving context and relationships.

## API Endpoints (Current Phase)

### Health & Status

- `GET /` - Welcome message and API info
- `GET /health` - Server health status
- `GET /config` - Current configuration (non-sensitive)

### Future Endpoints

- `POST /documents/upload` - Upload documents
- `POST /documents/process` - Process and chunk documents
- `GET /documents` - List uploaded documents
- `POST /query` - Query across documents
- `GET /query/{id}` - Get query history

## Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
black backend/
isort backend/
pylint backend/
```

### Debugging

- FastAPI interactive docs: http://localhost:8000/docs
- Check logs: `backend/logs/` (upcoming)

## Roadmap

- ✅ Phase 0: Project Foundation
- 📋 Phase 1-3: Document Processing Pipeline
- 📋 Phase 4-6: RAG Core
- 📋 Phase 7-9: Backend API & Frontend
- 📋 Phase 10-17: Advanced Features & Production

## Interview Preparation Topics

This project demonstrates:

- System design and architecture
- Document processing pipelines
- Vector databases and semantic search
- LLM integration and prompt engineering
- RAG fundamentals and evaluation
- FastAPI and async Python
- Full-stack application development
- Deployment and DevOps (Docker, K8s)
- Software engineering best practices

## Contributing

Contributions welcome! Please ensure:

- Code follows PEP 8 style guide
- All tests pass
- Documentation is updated
- Commit messages are descriptive

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:

- GitHub Issues: [Create an issue](https://github.com/yourusername/enterprise-rag-assistant/issues)
- Documentation: See `/docs` folder

## Acknowledgments

- OpenRouter for LLM API access
- ChromaDB for vector database
- Sentence Transformers for embeddings
- FastAPI for the excellent framework

---

**Current Phase**: Phase 0 - Project Foundation ✅
**Last Updated**: 2026-08-13
