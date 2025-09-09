"""
Simple logging configuration.
"""

import logging
import os

def setup_logging():
    """Setup basic logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app.log', mode='a') if os.path.exists('logs') else logging.StreamHandler()
        ]
    )

def get_logger(name: str):
    """Get a logger instance."""
    return logging.getLogger(name)