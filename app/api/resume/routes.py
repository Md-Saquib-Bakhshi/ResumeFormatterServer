import io
import re
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import asyncio
import os
import tempfile
import shutil # For cleaning up temporary directories
from typing import List, Tuple

# Import the service functions and job manager
from . import services
from ...utils.job_manager import job_manager, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload-resumes", status_code=status.HTTP_202_ACCEPTED)
async def upload_resumes(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Handles multiple resume uploads. Files are saved temporarily, and processing
    is delegated to a background task. Returns a job ID to poll for status.
    All successful outputs will be a ZIP file containing PDFs.
    """
    if not files:
        logger.warning("No files provided in the /upload-resumes request.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    allowed_content_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
        "application/msword" # .doc
    }

    files_to_process: List[Tuple[str, str, str]] = [] # (temp_file_path, original_filename, content_type)
    original_filenames: List[str] = []

    # Create a temporary directory for this specific job
    temp_dir = tempfile.mkdtemp(prefix="resume_upload_")
    logger.info(f"Created temporary upload directory for job: {temp_dir}")
    
    try:
        for file in files:
            if file.content_type not in allowed_content_types:
                logger.warning(f"Unsupported file type '{file.content_type}' for '{file.filename}' in batch upload.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail=f"Unsupported file type: {file.content_type} for '{file.filename}'. Only PDF, DOCX, and DOC are accepted."
                )
            
            original_filenames.append(file.filename)
            # Store in the job's dedicated temporary directory
            temp_file_path = os.path.join(temp_dir, file.filename) 
            
            # Use aiofiles to asynchronously write the file to temp storage
            async with aiofiles.open(temp_file_path, 'wb') as out_file:
                while content := await file.read(1024): # Read in chunks
                    await out_file.write(content)
            
            files_to_process.append((temp_file_path, file.filename, file.content_type))
            logger.info(f"File '{file.filename}' saved temporarily to '{temp_file_path}'.")

        # Create a new job with the manager, passing the temp_dir
        job_id = str(uuid.uuid4()) # Generate job_id here
        job_manager.create_job(job_id, original_filenames, temp_dir) # Pass job_id and temp_dir

        # Schedule the processing as a background task
        background_tasks.add_task(services.process_batch_of_resumes, job_id, files_to_process, temp_dir)
        
        logger.info(f"Batch processing job '{job_id}' initiated for {len(files)} files.")
        return JSONResponse(
            content={"job_id": job_id, "status": "processing_initiated", "message": "Your files are being processed in the background. Use the job_id to check status and download results."},
            status_code=status.HTTP_202_ACCEPTED
        )

    except HTTPException:
        # Re-raise HTTPException if it originated from content type check
        raise
    except Exception as e:
        logger.error(f"Error during initial file upload or job creation: {e}", exc_info=True)
        # Clean up temp directory if an error occurred before job was fully enqueued
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory {temp_dir} due to error.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to initiate file processing: {e}")


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Checks the status of a background processing job.
    """
    job = job_manager.get_job_status(job_id)
    if not job:
        logger.warning(f"Status requested for non-existent job ID: {job_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    logger.info(f"Status requested for job '{job_id}'. Current status: {job.status.value}")
    
    response_content = {
        "job_id": job_id,
        "status": job.status.value,
        "original_filenames": job.original_filenames,
        "progress": job.progress
    }

    if job.status == JobStatus.FAILED:
        response_content["error_message"] = job.error_message
    elif job.status == JobStatus.COMPLETED:
        response_content["message"] = "Processing completed successfully. Download the ZIP file using the /download/{job_id} endpoint."
    
    return JSONResponse(content=response_content)


@router.get("/download/{job_id}")
async def download_job_results(
    job_id: str,
    background_tasks: BackgroundTasks
):
    """
    Downloads the processed results for a job as a ZIP file containing PDFs.
    """
    job_data = job_manager.get_job_status(job_id)

    if not job_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    if job_data.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is still processing. Please wait.")
    
    if job_data.status == JobStatus.FAILED:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Job failed: {job_data.error_message}")

    # This is now the ONLY logic for completed jobs: serving the ZIP file
    if job_data.status == JobStatus.COMPLETED and job_data.zip_file_path:
        zip_file_path = job_data.zip_file_path
        if not os.path.exists(zip_file_path):
            logger.error(f"ZIP file path not found for job '{job_id}': {zip_file_path}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result ZIP file not found.")

        async def file_iterator():
            async with aiofiles.open(zip_file_path, mode="rb") as f:
                while chunk := await f.read(8192): # Read in chunks
                    yield chunk

        # Define a cleanup task for the zip file and the temp upload directory after it has been sent
        async def cleanup_job_files_async():
            # Small delay to ensure FastAPI has flushed the response before deleting files
            await asyncio.sleep(0.1) 
            
            # Clean up the generated ZIP file
            if os.path.exists(zip_file_path):
                try:
                    os.remove(zip_file_path)
                    logger.info(f"Cleaned up ZIP file: {zip_file_path}")
                except OSError as e:
                    logger.error(f"Error removing ZIP file {zip_file_path}: {e}")
            
            # Clean up the temporary upload directory for this job
            if job_data.temp_dir and os.path.exists(job_data.temp_dir):
                try:
                    shutil.rmtree(job_data.temp_dir)
                    logger.info(f"Cleaned up temporary upload directory: {job_data.temp_dir}")
                except OSError as e:
                    logger.error(f"Error removing temporary upload directory {job_data.temp_dir}: {e}")

        # Add the cleanup task to background tasks
        background_tasks.add_task(cleanup_job_files_async)
        
        logger.info(f"Serving ZIP file '{zip_file_path}' for job '{job_id}'.")
        return StreamingResponse(
            file_iterator(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=\"parsed_resumes_{job_id}_pdfs.zip\""}
        )

    # Fallback if somehow completed but no zip_file_path (shouldn't happen with current logic)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job results not ready or invalid state.")