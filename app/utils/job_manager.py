import uuid
import logging
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

class JobStatus:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class JobManager:
    """
    A simple in-memory manager for background job states.
    NOTE: This is NOT persistent. Data will be lost on application restart.
    For production, consider Redis, a database, or a dedicated task queue like Celery.
    """
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {} # {job_id: {status: str, results: Any, error: str, ...}}
        logger.info("JobManager initialized (in-memory).")

    def create_job(self, original_filenames: list[str]) -> str:
        """Creates a new job and returns its ID."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "original_filenames": original_filenames,
            "results": [], # List of parsed JSON results for each file
            "zip_file_path": None, # Path to the generated zip file
            "error_message": None,
            "progress": 0 # Optional: 0-100 percentage
        }
        logger.info(f"Job '{job_id}' created for files: {original_filenames}")
        return job_id

    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Updates the status and other attributes of a job."""
        if job_id not in self._jobs:
            logger.warning(f"Attempted to update non-existent job ID: {job_id}")
            return False
        
        job = self._jobs[job_id]
        job["status"] = status
        for key, value in kwargs.items():
            job[key] = value
        
        logger.info(f"Job '{job_id}' status updated to: {status}. Additional data: {kwargs}")
        return True

    def get_job(self, job_id: str) -> Union[Dict[str, Any], None]:
        """Retrieves job details by ID."""
        return self._jobs.get(job_id)

    def delete_job(self, job_id: str):
        """Deletes a job from memory."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info(f"Job '{job_id}' deleted from memory.")
        else:
            logger.warning(f"Attempted to delete non-existent job ID: {job_id}")


job_manager = JobManager() # Global instance