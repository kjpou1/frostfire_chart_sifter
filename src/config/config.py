import os
from typing import Optional

import tensorflow as tf
from dotenv import load_dotenv

from src.models import SingletonMeta


class Config(metaclass=SingletonMeta):
    """
    Singleton configuration class for managing project-wide constants and directories.

    This class handles environment variables, default values, and ensures that
    necessary directories are created for the application.
    """

    _is_initialized = False  # Tracks if the Config has already been initialized

    def __init__(self):
        """
        Initialize the configuration class. Loads environment variables, sets defaults,
        and creates necessary directories. Prevents re-initialization if already initialized.
        """
        # Prevent re-initialization
        if Config._is_initialized:
            return

        # Load environment variables from .env file
        load_dotenv()

        # Set constants from environment variables or use defaults
        self.IMG_SIZE = tuple(
            map(int, os.getenv("IMG_SIZE", "224,224").split(","))
        )  # Image size (height, width)
        self.INPUT_SHAPE = (*self.IMG_SIZE, 3)  # Input shape for TensorFlow models
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))  # Batch size for training
        self.EPOCHS = int(os.getenv("EPOCHS", 35))  # Number of training epochs
        self._MODEL_TYPE = os.getenv("MODEL_TYPE", None)  # Model type to be used

        # Dataset buffer sizes for shuffling and prefetching
        self.SHUFFLE_BUFFER_SIZE = int(
            os.getenv("SHUFFLE_BUFFER_SIZE", 1000)
        )  # Shuffle buffer size
        self.PREFETCH_BUFFER_SIZE = (
            tf.data.AUTOTUNE
        )  # Auto-tune prefetch buffer size for optimal performance

        # Base directory for all artifacts
        self.BASE_DIR = os.getenv("BASE_DIR", "artifacts")

        # Subdirectories for different types of artifacts
        self.RAW_DATA_DIR = os.path.join(
            self.BASE_DIR, "data", "raw"
        )  # Raw data storage
        self.PROCESSED_DATA_DIR = os.path.join(
            self.BASE_DIR, "data", "processed"
        )  # Processed data storage
        self.FEATURES_DIR = os.path.join(
            self.BASE_DIR, "data", "features"
        )  # Feature data storage
        self.MODEL_DIR = os.path.join(
            self.BASE_DIR, "models"
        )  # Trained models directory
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")  # Logs directory
        self.METADATA_DIR = os.path.join(
            self.BASE_DIR, "metadata"
        )  # Metadata directory
        self.METADATA_FILE_PATH = os.path.join(
            self.METADATA_DIR, "metadata.json"
        )  # Metadata file path
        self.REPORTS_DIR = os.path.join(self.BASE_DIR, "reports")  # Reports directory
        self.HISTORY_DIR = os.path.join(
            self.BASE_DIR, "history"
        )  # Training history directory

        # Ensure all required directories exist
        self._ensure_directories_exist()

        # Mark the Config as initialized
        Config._is_initialized = True

    def _ensure_directories_exist(self):
        """
        Ensure that all necessary directories exist. Creates directories if they do not exist.

        Raises:
            OSError: If directory creation fails.
        """
        directories = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.FEATURES_DIR,
            self.MODEL_DIR,
            self.LOG_DIR,
            self.METADATA_DIR,
            self.REPORTS_DIR,
            self.HISTORY_DIR,
        ]
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                raise OSError(f"Failed to create directory {directory}: {e}")

    @property
    def model_type(self) -> str:
        """
        Getter for the model type.

        Returns:
            str: The current model type.
        """
        return self._MODEL_TYPE

    @model_type.setter
    def model_type(self, value: Optional[str]) -> None:
        """
        Setter for the model type.

        Args:
            value (Optional[str]): The new model type to be set. Can be None.

        Raises:
            ValueError: If the value is not a string when it is not None.
        """
        if value is not None and not isinstance(value, str):
            raise ValueError("Model type must be a string or None.")
        self._MODEL_TYPE = value

    @classmethod
    def initialize(cls):
        """
        Explicitly initialize the Config singleton if not already initialized.
        """
        if not cls._is_initialized:
            cls()

    @classmethod
    def reset(cls):
        """
        Reset the Config singleton. Useful for testing purposes.
        """
        cls._is_initialized = False
        cls._instances = {}
