"""
Logging utilities for training.
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(exp_name: str, output_dir: Path) -> logging.Logger:
    """
    Setup logger for experiment.
    
    Args:
        exp_name: experiment name
        output_dir: output directory
        
    Returns:
        configured logger
    """
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
