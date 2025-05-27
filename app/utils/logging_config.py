import logging
import os

def setup_logging():
    """
    Sets up the application-wide logging configuration.
    Logs will be written to 'app.log' in the project root.
    The log file will be truncated (cleaned) on each application start.
    """
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'app.log')
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set default logging level

    # Clear existing handlers to prevent duplicate logs (important for reloader)
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Create a file handler
    # Change mode from 'a' (append) to 'w' (write) to truncate the file on each run
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO) # Minimum level for file output

    # Create a console handler (for seeing logs in terminal during development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Minimum level for console output

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Suppress verbose logs from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)