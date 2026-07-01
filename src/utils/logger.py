import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO):
    """Setup logging configuration"""
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                
                # Ensure logs directory exists
                log_filename = config.get('handlers', {}).get('file', {}).get('filename')
                if log_filename:
                    log_dir = os.path.dirname(log_filename)
                    if log_dir:
                        os.makedirs(log_dir, exist_ok=True)

                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error loading logging config: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print(f"Failed to load configuration file {path}. Using basic config.")

def get_logger(name: str) -> logging.Logger:
    """Get a logger by name"""
    return logging.getLogger(name)
