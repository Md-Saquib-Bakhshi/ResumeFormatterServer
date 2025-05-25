import os
import uuid
import logging
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class JobData:
    """
    Data structure to hold information about a processing job.
    """
    def __init__(
        self,
        job_id: str,
        status: JobStatus = JobStatus.PENDING,
        progress: int = 0,
        original_filenames: List[str] = None,
        results: List[Dict[str, Any]] = None,
        error_message: str = None,
        zip_file_path: str = None,
        temp_dir: str = None # Added for temporary upload directory cleanup
    ):
        self.job_id = job_id
        self.status = status
        self.progress = progress
        self.original_filenames = original_filenames if original_filenames is not None else []
        self.results = results if results is not None else []
        self.error_message = error_message
        self.zip_file_path = zip_file_path
        self.temp_dir = temp_dir # Store the temporary directory path

class JobManager:
    """
    Manages the state and data for all ongoing and completed jobs.
    """
    def __init__(self):
        self.jobs: Dict[str, JobData] = {}
        logger.info("JobManager initialized.")

    def create_job(self, job_id: str, original_filenames: List[str], temp_dir: str):
        """
        Creates a new job entry with initial PENDING status.
        'temp_dir' is the path to the temporary directory where files for this job are stored.
        """
        if job_id in self.jobs:
            logger.warning(f"Attempted to create job {job_id} which already exists. Overwriting.")
        
        self.jobs[job_id] = JobData(
            job_id=job_id,
            status=JobStatus.PENDING,
            original_filenames=original_filenames,
            results=[], # Initialize with an empty list for results
            temp_dir=temp_dir # Store the temp directory here
        )
        logger.info(f"Job '{job_id}' created for files: {original_filenames}, temp_dir: {temp_dir}")
        return job_id # Return the job_id for convenience

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[int] = None,
        results: Optional[List[Dict[str, Any]]] = None,
        error_message: Optional[str] = None,
        zip_file_path: Optional[str] = None,
        temp_dir: Optional[str] = None # Added for potential updates, though primarily set at creation
    ):
        """
        Updates the status and other data for an existing job.
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found for status update.")
            # For a background task, raising an HTTPException might not be caught by FastAPI,
            # so logging and returning is more appropriate than raising a ValueError directly here.
            # If this is called from a FastAPI endpoint, then HTTPException is fine.
            raise ValueError(f"Job {job_id} not found for status update.")

        job = self.jobs[job_id]
        job.status = status
        if progress is not None:
            job.progress = progress
        if results is not None:
            job.results = results
        if error_message is not None:
            job.error_message = error_message
        if zip_file_path is not None:
            job.zip_file_path = zip_file_path
        if temp_dir is not None:
            job.temp_dir = temp_dir
        
        logger.debug(f"Job '{job_id}' updated: Status={status.value}, Progress={progress}")


    def get_job_status(self, job_id: str) -> Optional[JobData]:
        """
        Retrieves the current status and data for a job.
        """
        return self.jobs.get(job_id)

# Global instance of JobManager
job_manager = JobManager()