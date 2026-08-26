"""
Main FastAPI application for Enterprise RAG Assistant
"""
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .api.chat import router as chat_router
from .api.documents import router as documents_router
from .api.errors import register_error_handlers
from .api.health import router as health_router
from .middleware.request_id import RequestIDMiddleware
from .middleware.request_logging import RequestLoggingMiddleware

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Multi-Document RAG Assistant",
    description=(
        "Multi-document retrieval-augmented generation (RAG) API for "
        "enterprise document analysis."
    ),
    version="0.1.0"
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request-ID middleware (Step 7.10.2) – inner, sets request.state.request_id
app.add_middleware(RequestIDMiddleware)

# Request logging middleware (Step 7.10.3) – outer, measures full request duration
app.add_middleware(RequestLoggingMiddleware)

# Global error-response contract (Step 7.10.1)
register_error_handlers(app)

# Pydantic models for API responses
class WelcomeResponse(BaseModel):
    message: str
    api_version: str
    documentation_url: str

# Routes
@app.get("/", response_model=WelcomeResponse)
async def root():
    """Welcome endpoint"""
    return WelcomeResponse(
        message="Welcome to Enterprise Multi-Document RAG Assistant",
        api_version="0.1.0",
        documentation_url="/docs"
    )

app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api/documents")

@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive info)"""
    return {
        "debug": os.getenv("DEBUG", "False"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        "max_tokens": os.getenv("LLM_MAX_TOKENS", "2048"),
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup initiated")
    logger.info(f"Debug mode: {os.getenv('DEBUG', 'False')}")
    logger.info(f"LLM Model: {os.getenv('LLM_MODEL', 'Not configured')}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "False") == "True"
    )
