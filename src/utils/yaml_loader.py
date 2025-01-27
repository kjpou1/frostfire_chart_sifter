import os

import yaml

from src.exception import CustomException
from src.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


def load_model_config(file_path="config/model_config.yaml"):
    """
    Load the YAML configuration file for models.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed YAML configuration as a dictionary.

    Raises:
        CustomException: If there's an issue loading the file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info("Model configuration loaded successfully.")
        return config
    except Exception as e:
        logging.error(f"Failed to load model configuration: {e}")
        raise CustomException(e)
