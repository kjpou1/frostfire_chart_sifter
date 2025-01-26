import os

import tensorflow as tf
from dotenv import load_dotenv

from src.models import SingletonMeta


class Config(metaclass=SingletonMeta):
    """
    Singleton configuration class for managing project-wide constants and directories.
    """

    _is_initialized = False  # Tracks whether the Config has already been initialized

    def __init__(self):
        # Prevent re-initialization
        if Config._is_initialized:
            return

        # Load environment variables
        load_dotenv()

        # Constants
        self.IMG_SIZE = (224, 224)  # Image size (height, width)
        self.BATCH_SIZE = 32  # Batch size

        # Dataset Buffer Sizes
        self.SHUFFLE_BUFFER_SIZE = 1000
        self.PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

        # Base directory for artifacts
        self.BASE_DIR = os.getenv("BASE_DIR", "artifacts")

        # Subdirectories for artifacts
        self.RAW_DATA_DIR = os.path.join(self.BASE_DIR, "data", "raw")
        self.PROCESSED_DATA_DIR = os.path.join(self.BASE_DIR, "data", "processed")
        self.FEATURES_DIR = os.path.join(self.BASE_DIR, "data", "features")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.METADATA_DIR = os.path.join(self.BASE_DIR, "metadata")
        self.METADATA_FILE_PATH = os.path.join(self.METADATA_DIR, "metadata.json")
        self.REPORTS_DIR = os.path.join(self.BASE_DIR, "reports")

        # Ensure all necessary directories exist
        self._ensure_directories_exist()

        # Mark as initialized
        Config._is_initialized = True

    def _ensure_directories_exist(self):
        """
        Ensures that all necessary directories exist. Creates them if they do not.
        """
        directories = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.FEATURES_DIR,
            self.MODEL_DIR,
            self.LOG_DIR,
            self.METADATA_DIR,
            self.REPORTS_DIR,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def initialize(cls):
        """
        Explicitly initializes the Config singleton.
        This ensures that the configuration is set up before being used in the application.
        """
        if not cls._is_initialized:
            cls()

    @classmethod
    def is_initialized(cls):
        """
        Checks whether the Config singleton has been initialized.
        Returns:
            bool: True if initialized, False otherwise.
        """
        return cls._is_initialized

    @classmethod
    def reset(cls):
        """
        Resets the Config singleton for testing purposes.
        """
        cls._is_initialized = False
        cls._instances = {}
