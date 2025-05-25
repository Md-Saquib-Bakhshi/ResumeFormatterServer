import io
import re
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import logging
import asyncio
import os
import tempfile
from typing import List, Tuple, Literal

# Import the service functions and job manager
from . import services
from ...utils.job_manager import job_manager, JobStatus

# Import the new generation functions from their dedicated files
from .generators.docx_generator import generate_docx_from_json
from .generators.pdf_generator import generate_pdf_from_json

logger = logging.getLogger(__name__) # Get a logger instance for this module

router = APIRouter()

@router.post("/upload-resumes", status_code=202) # Return 202 Accepted for background processing
async def upload_resumes(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Handles multiple resume uploads. Files are saved temporarily, and processing
    is delegated to a background task. Returns a job ID to poll for status.
    """
    if not files:
        logger.warning("No files provided in the /upload-resumes request.")
        raise HTTPException(status_code=400, detail="No files provided.")

    allowed_content_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
        "application/msword" # .doc
    }

    # Store file info and paths for background task
    files_to_process: List[Tuple[str, str, str]] = [] # (temp_file_path, original_filename, content_type)
    original_filenames: List[str] = []

    temp_dir = tempfile.mkdtemp() # Create a temporary directory for this job
    
    try:
        for file in files:
            if file.content_type not in allowed_content_types:
                logger.warning(f"Unsupported file type '{file.content_type}' for '{file.filename}' in batch upload.")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.content_type} for '{file.filename}'. Only PDF, DOCX, and DOC are accepted."
                )
            
            original_filenames.append(file.filename)
            temp_file_path = os.path.join(temp_dir, file.filename) # Store in job's temp dir
            
            # Use aiofiles to asynchronously write the file to temp storage
            async with aiofiles.open(temp_file_path, 'wb') as out_file:
                while content := await file.read(1024): # Read in chunks
                    await out_file.write(content)
            
            files_to_process.append((temp_file_path, file.filename, file.content_type))
            logger.info(f"File '{file.filename}' saved temporarily to '{temp_file_path}'.")

        # Create a new job with the manager
        job_id = job_manager.create_job(original_filenames)

        # Schedule the processing as a background task
        # IMPORTANT: This uses asyncio.create_task which is not persistent.
        # For production, replace with Celery/RQ task dispatch.
        background_tasks.add_task(services.process_batch_of_resumes, job_id, files_to_process)
        
        logger.info(f"Batch processing job '{job_id}' initiated for {len(files)} files.")
        return JSONResponse(
            content={"job_id": job_id, "status": "processing_initiated", "message": "Your files are being processed in the background. Use the job_id to check status and download results."},
            status_code=202
        )

    except HTTPException:
        # Re-raise HTTPException if it originated from content type check
        raise
    except Exception as e:
        logger.error(f"Error during initial file upload or job creation: {e}", exc_info=True)
        # Clean up temp directory if an error occurred before job was fully enqueued
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory {temp_dir} due to error.")
        raise HTTPException(status_code=500, detail=f"Failed to initiate file processing: {e}")


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Checks the status of a background processing job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        logger.warning(f"Status requested for non-existent job ID: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found.")

    logger.info(f"Status requested for job '{job_id}'. Current status: {job['status']}")
    
    response_content = {
        "job_id": job_id,
        "status": job["status"],
        "original_filenames": job["original_filenames"],
        "progress": job.get("progress", 0)
    }

    if job["status"] == JobStatus.FAILED:
        response_content["error_message"] = job["error_message"]
    elif job["status"] == JobStatus.COMPLETED:
        response_content["message"] = "Processing completed successfully."
        # Do not return results directly here, they are downloaded via /download/{job_id}
    
    return JSONResponse(content=response_content)


@router.get("/download/{job_id}")
async def download_job_results(
    job_id: str,
    format: Literal["json", "pdf", "docx"] = "json" # Added format query parameter
):
    """
    Downloads the processed results for a completed job.
    Returns a ZIP file for multiple files, or a single JSON/PDF/DOCX for one file.
    """
    job = job_manager.get_job(job_id)
    if not job:
        logger.warning(f"Download requested for non-existent job ID: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found.")

    if job["status"] != JobStatus.COMPLETED:
        logger.info(f"Download requested for job '{job_id}' but status is '{job['status']}'.")
        raise HTTPException(
            status_code=409, # Conflict
            detail=f"Job '{job_id}' is not yet completed. Current status: {job['status']}."
        )

    # Handle multi-file jobs (always return ZIP of JSONs)
    if len(job["original_filenames"]) > 1:
        if format != "json":
            logger.warning(f"Format '{format}' requested for multi-file job '{job_id}'. Returning ZIP of JSONs by default.")
        
        if job.get("zip_file_path") and os.path.exists(job["zip_file_path"]):
            zip_path = job["zip_file_path"]
            logger.info(f"Serving ZIP file for job '{job_id}' from path: {zip_path}")

            # Define the cleanup function as async
            async def cleanup_zip_async():
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    logger.info(f"Cleaned up temporary ZIP file: {zip_path}")
            
            response = FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename=f"RESUME_PARSE_RESULTS_{job_id}.zip"
            )
            # Assign the async cleanup function to background tasks
            response.background = BackgroundTasks([cleanup_zip_async])
            return response
        else:
            logger.error(f"Job '{job_id}' completed (multi-file), but no ZIP file path found or file missing.")
            raise HTTPException(status_code=500, detail="Results ZIP file not found for multi-file job.")
    
    # Handle single-file jobs
    elif len(job["results"]) == 1 and job["results"][0].get("parsed_data"):
        single_result = job["results"][0]["parsed_data"]
        original_fname = job["results"][0].get("original_filename", "parsed_result")
        
        extracted_name = single_result.get("basic_details", {}).get("name", "").strip()
        
        # Determine base filename for download (e.g., "JOHN_DOE_RESUME" or "PARSED_MY_FILE_RESUME")
        if extracted_name and extracted_name != "N/A":
            sanitized_name_upper = re.sub(r'[^\w\s-]', '', extracted_name).strip().replace(' ', '_').upper()
            base_filename = f"{sanitized_name_upper}_RESUME"
        else:
            name_part = os.path.splitext(original_fname)[0]
            sanitized_name_part = re.sub(r'[^\w\s-]', '', name_part).strip().replace(' ', '_').upper()
            base_filename = f"PARSED_{sanitized_name_part}_RESUME"

        if format == "json":
            output_filename = f"{base_filename}.json"
            logger.info(f"Serving single JSON result for job '{job_id}' as '{output_filename}'.")
            return JSONResponse(content=single_result, media_type="application/json", headers={"Content-Disposition": f"attachment; filename=\"{output_filename}\""})
        
        elif format == "pdf":
            try:
                # Run synchronous PDF generation in a separate thread to avoid blocking the event loop
                pdf_bytes = await asyncio.to_thread(generate_pdf_from_json, single_result)
                output_filename = f"{base_filename}.pdf"
                logger.info(f"Serving single PDF result for job '{job_id}' as '{output_filename}'.")
                return FileResponse(
                    content=io.BytesIO(pdf_bytes), # FileResponse expects a file-like object or path
                    media_type="application/pdf",
                    filename=output_filename,
                    headers={"Content-Length": str(len(pdf_bytes))} # Essential for proper download
                )
            except Exception as e:
                logger.error(f"Failed to generate PDF for job '{job_id}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")

        elif format == "docx":
            try:
                # Run synchronous DOCX generation in a separate thread
                docx_bytes = await asyncio.to_thread(generate_docx_from_json, single_result)
                output_filename = f"{base_filename}.docx"
                logger.info(f"Serving single DOCX result for job '{job_id}' as '{output_filename}'.")
                return FileResponse(
                    content=io.BytesIO(docx_bytes),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    filename=output_filename,
                    headers={"Content-Length": str(len(docx_bytes))}
                )
            except Exception as e:
                logger.error(f"Failed to generate DOCX for job '{job_id}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to generate DOCX: {e}")

    else:
        logger.error(f"Job '{job_id}' completed, but no downloadable results found or format unexpected for single file.")
        raise HTTPException(status_code=500, detail="Results not available in expected format for download.")