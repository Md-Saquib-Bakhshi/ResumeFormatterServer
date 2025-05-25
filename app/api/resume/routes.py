from fastapi import APIRouter, UploadFile, File, HTTPException
import logging

# Import the service function
from . import services

logger = logging.getLogger(__name__) # Get a logger instance for this module

router = APIRouter()

@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Handles the resume upload, now supporting various document types (PDF, DOC, DOCX).
    Processes based on file type (direct text extraction for Word docs, PDF/OCR for PDFs),
    delegates processing to the service layer, and returns the parsed resume data
    with a dynamic filename.
    """
    ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
        "application/msword" # .doc
    }

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Received upload with unsupported file type: '{file.content_type}' for file '{file.filename}'.")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Only PDF, DOCX, and DOC are accepted."
        )

    try:
        file_bytes = await file.read()
        
        logger.info(f"Starting processing for uploaded file: '{file.filename}' (Type: {file.content_type}).")
        result = await services.process_resume_document(file_bytes, file.filename, file.content_type)
        logger.info(f"Successfully processed file: '{file.filename}'. Output filename: '{result['filename']}'.")
        
        return {
            "filename": result["filename"],
            "parsed_resume_data": result["parsed_resume_data"]
        }

    except Exception as e:
        logger.error(f"Fatal error during resume processing for file '{file.filename}': {e}", exc_info=True)
        if "Azure OpenAI client is not initialized" in str(e):
             raise HTTPException(
                status_code=500,
                detail="LLM service is not configured. Please ensure Azure OpenAI environment variables are set."
             )
        elif "Could not read DOCX file" in str(e) or "Could not read DOC file" in str(e):
            raise HTTPException(
                status_code=422, # Unprocessable Entity
                detail=f"Failed to read or parse the document content: {e}. Please ensure the file is not corrupted."
            )
        else:
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")