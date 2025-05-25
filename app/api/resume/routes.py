from fastapi import APIRouter, UploadFile, File, HTTPException

# Import the service function
from . import services

router = APIRouter()

@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Handles the resume upload, now supporting various document types (PDF, DOC, DOCX).
    Processes based on file type (direct text extraction for Word docs, PDF/OCR for PDFs),
    delegates processing to the service layer, and returns the parsed resume data
    with a dynamic filename.
    """
    # Define allowed content types
    ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
        "application/msword" # .doc 
    }

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Only PDF, DOCX, and DOC are accepted."
        )

    try:
        file_bytes = await file.read()
        
        # Delegate the document processing to the service layer, passing content_type
        result = await services.process_resume_document(file_bytes, file.filename, file.content_type)
        
        return {
            "filename": result["filename"],
            "parsed_resume_data": result["parsed_resume_data"]
        }

    except Exception as e:
        print(f"FATAL ERROR during resume processing: {e}")
        # Provide a general internal server error message
        if "Azure OpenAI client is not initialized" in str(e):
             raise HTTPException(
                status_code=500,
                detail="LLM service is not configured. Please ensure Azure OpenAI environment variables are set."
             )
        else:
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")