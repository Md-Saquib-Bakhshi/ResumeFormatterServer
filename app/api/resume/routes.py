from fastapi import APIRouter, UploadFile, File, HTTPException

# Import the service function
from . import services

router = APIRouter()

@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Handles the resume upload, delegates processing to the service layer,
    and returns the parsed resume data with a dynamic filename.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        pdf_bytes = await file.read()
        
        # Delegate the core processing logic to the service layer
        result = await services.process_resume_pdf(pdf_bytes)
        
        # Return the simplified response as requested
        return {
            "filename": result["filename"],
            "parsed_resume_data": result["parsed_resume_data"]
        }

    except Exception as e:
        print(f"FATAL ERROR during resume processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")