import logging
import sys

def setup_logger(name: str = "scLightGAT"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add formatter to console handler
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

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

