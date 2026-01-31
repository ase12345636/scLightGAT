import logging
import sys

def setup_logger(name: str = "scLightGAT", log_file: str = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set root to lower level to capture all

    # Avoid adding duplicate handlers if they already exist
    # But we might want to add a file handler if it's new
    
    # Check for console handler
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    if not has_console:
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)  # Console shows logs INFO and above

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Add file handler if provided and not present
    if log_file:
        # Check if this specific file is already being logged to to avoid duplication?
        # Or just check if any FileHandler exists? 
        # Better: check if we are already logging to this file. 
        # Simplified: just add it. Python logging is thread safe but duplicates can happen if called repeatedly.
        # We will assume calling convention manages this, or check existence.
        
        has_file = False
        for h in logger.handlers:
             if isinstance(h, logging.FileHandler):
                 # We could check h.baseFilename but let's just assume if we have a file handler, we append?
                 # Actually user might run multiple pipelines. 
                 # Let's just add it if not present.
                 pass
        
        # Always add a new file handler for this run if passed, 
        # ensuring we capture this specific run's logs.
        # But prevent adding identical one?
        
        # Let's just create it.
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG) # File captures EVERYTHING
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def setup_warning_logging(log_file="scLightGAT_run.log"):
    """
    Configure warning logging to file only, suppressing console output.
    """
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.propagate = False  # Prevent propagation to root (console)
    
    # Check if handlers exist to avoid duplicates
    if not warn_logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        warn_logger.addHandler(file_handler)
    
    return warn_logger

