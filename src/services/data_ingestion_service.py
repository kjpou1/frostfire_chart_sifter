import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.models.data_ingestion_config import DataIngestionConfig
from src.services.dataset_splitter_service import DatasetSplitterService
from src.services.hugging_face_service import HuggingFaceService
from src.utils.file_utils import load_datasets_from_directory

logging = LoggerManager.get_logger(__name__)


class DataIngestionService:
    """
    A class for handling the data ingestion process.
    Reads input data, splits it into training and testing datasets, and saves them as CSV files.
    """

    def __init__(self):
        """
        Initializes the DataIngestion class by creating an instance of the DataIngestionConfig.
        """
        self.ingestion_config = DataIngestionConfig()
        self.huggingface_service = HuggingFaceService()
        self.dataset_splitter_service = DatasetSplitterService()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = Config().BATCH_SIZE
        self.img_size = Config().IMG_SIZE

    async def initiate_data_ingestion(self, test_size=0.2):
        """
        Orchestrates the entire data ingestion process:
        1. Downloads datasets using HuggingFaceService.
        2. Splits datasets into train, validation, and test sets using DatasetSplitterService.
        3. Loads datasets into TensorFlow-friendly format for model training.

        Args:
            test_size (float): Proportion of the dataset to include in the test split (default is 0.2).

        Returns:
            tuple: Train, validation, and test datasets as TensorFlow datasets.

        Raises:
            CustomException: If any error occurs during the data ingestion process.
        """
        logging.info("Starting the data ingestion process...")
        try:
            # Step 1: Download datasets from HuggingFace or other sources
            logging.info("Initiating dataset download via HuggingFaceService...")
            self.huggingface_service.initiate_download()
            logging.info("Dataset download completed.")

            # Step 2: Split datasets into train, validation, and test sets
            logging.info("Splitting datasets into train, validation, and test sets...")
            self.dataset_splitter_service.initiate_split()
            logging.info("Dataset splitting completed.")

            # Step 3: Load datasets into TensorFlow-friendly format
            logging.info("Loading datasets into TensorFlow-friendly format...")
            await self.initiate_data_ingestion_tf()
            logging.info("Datasets loaded successfully.")

            logging.info("Data ingestion process completed.")
            return self.train_dataset, self.val_dataset, self.test_dataset

        except Exception as e:
            logging.error("Error occurred during data ingestion: %s", str(e))
            raise CustomException(e, sys) from e

    async def initiate_data_ingestion_tf(self):
        """
        Loads datasets from the split directories into TensorFlow-friendly format.
        Uses batch size and image size configurations to prepare the datasets.
        """
        try:
            logging.info(
                "Loading training dataset from directory: %s",
                self.ingestion_config.train_data_dir,
            )
            self.train_dataset = load_datasets_from_directory(
                self.ingestion_config.train_data_dir,
                batch_size=self.batch_size,
                img_size=self.img_size,
            )
            logging.info("Training dataset loaded successfully.")

            logging.info(
                "Loading validation dataset from directory: %s",
                self.ingestion_config.val_data_dir,
            )
            self.val_dataset = load_datasets_from_directory(
                self.ingestion_config.val_data_dir,
                batch_size=self.batch_size,
                img_size=self.img_size,
            )
            logging.info("Validation dataset loaded successfully.")

            logging.info(
                "Loading test dataset from directory: %s",
                self.ingestion_config.test_data_dir,
            )
            self.test_dataset = load_datasets_from_directory(
                self.ingestion_config.test_data_dir,
                batch_size=self.batch_size,
                img_size=self.img_size,
            )
            logging.info("Test dataset loaded successfully.")

        except Exception as e:
            logging.error(
                "Error occurred during TensorFlow dataset ingestion: %s", str(e)
            )
            raise CustomException(e, sys) from e


# if __name__ == "__main__":
#     """
#     Entry point for the script.
#     Initializes the DataIngestion class and executes the data ingestion process.
#     """
#     obj = DataIngestion()
#     obj.initiate_data_ingestion()
