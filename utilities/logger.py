import logging
import os
def setup_logger(name, log_file, level=logging.DEBUG):  # Change default to DEBUG
    """
    Configures a logger with the specified name and log file.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Explicitly set to DEBUG

    # Prevent duplicate handlers
    if not logger.handlers:
        # Create a file handler for logging to a file
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Set file handler to DEBUG

        # Create a console handler for output to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Set console handler to DEBUG

        # Define a formatter and add it to both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger