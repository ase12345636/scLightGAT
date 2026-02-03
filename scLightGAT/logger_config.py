import logging
import sys

def setup_logger(name: str = "scLightGAT", log_file: str = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set root to lower level to capture all

    # Check if this logger or its parents already have handlers
    # If a parent has handlers and propagate is True, we don't need to add a console handler
    has_console = False
    
    # Check current logger
    if any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        has_console = True
    
    # Check parents if propagation is on
    if logger.propagate and not has_console:
        parent = logger.parent
        while parent:
            if any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in parent.handlers):
                has_console = True
                break
            if not parent.propagate:
                break
            parent = parent.parent

    # Only add console handler if NO parent handles it (i.e. we are setting up the root or a detached logger)
    # OR if this is the main setup call (name="scLightGAT")
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

    # Add file handler if provided
    if log_file:
        # Check if we already have a file handler for THIS specific file
        # This is a bit tricky, but we can check baseFilename of FileHandlers
        
        # Resolve absolute path to be sure
        import os
        abs_log_file = os.path.abspath(log_file)
        
        has_file = False
        for h in logger.handlers:
             if isinstance(h, logging.FileHandler):
                 if os.path.abspath(h.baseFilename) == abs_log_file:
                     has_file = True
                     break
        
        if not has_file:
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

