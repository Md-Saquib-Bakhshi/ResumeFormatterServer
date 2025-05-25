from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.resume.routes import router as resume_router
import logging
from .utils.logging_config import setup_logging # Import the logging setup function

# Set up logging as the very first step
setup_logging()
logger = logging.getLogger(__name__) # Get a logger instance for this module

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:5173", 
                   "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume_router, prefix="/resume", tags=["resume"])

@app.on_event("startup")
def on_startup():
    logger.info("🚀 Server is running... Visit http://localhost:8000/docs")