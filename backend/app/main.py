"""
Main FastAPI application for Enterprise RAG Assistant
"""
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise RAG Assistant",
    description="A multi-document RAG system for enterprise document analysis",
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

# Pydantic models for API responses
class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str

class WelcomeResponse(BaseModel):
    message: str
    api_version: str
    documentation_url: str

# Routes
@app.get("/", response_model=WelcomeResponse)
async def root():
    """Welcome endpoint"""
    return WelcomeResponse(
        message="Welcome to Enterprise RAG Assistant",
        api_version="0.1.0",
        documentation_url="/docs"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.info("Health check performed")
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        environment=os.getenv("DEBUG", "False")
    )

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
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "False") == "True"
    )
