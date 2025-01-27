import os
import sys
import uuid
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.services.data_ingestion_service import DataIngestionService
from src.services.data_transformation_service import DataTransformationService
from src.services.dataset_splitter_service import DatasetSplitterService
from src.services.hugging_face_service import HuggingFaceService
from src.services.model_selection_service import ModelSelectionService
from src.services.model_training_service import ModelTrainingService
from src.utils.file_utils import save_json, save_object, save_training_artifacts
from src.utils.ml_utils import create_model, create_model_from_config

logging = LoggerManager.get_logger(__name__)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_service = DataIngestionService()
        self.data_transformation_service = DataTransformationService()
        # self.model_selection_service = ModelSelectionService()
        self.model_training_service = ModelTrainingService()
        self.config = Config()

    async def run_pipeline(self):
        try:

            # Step 1: Data Ingestion
            logging.info("Starting data ingestion.")
            # train_path, test_path = (
            #     self.data_ingestion_service.initiate_data_ingestion()
            # )
            train_tf_dataset, val_tf_dataset, test_tf_dataset = (
                await self.data_ingestion_service.initiate_data_ingestion()
            )
            logging.info(
                f"Data ingested. \nTrain: {train_tf_dataset}, \nVal: {val_tf_dataset}, \nTest: {train_tf_dataset}"
            )
            # Example: Inspect a batch
            for images, labels in train_tf_dataset.take(1):
                logging.info(f"Batch of images shape: {images.shape}")
                logging.info(f"Batch of labels: {labels}")

            # Get one batch of data
            sample_batch = list(train_tf_dataset.take(1))[0]

            # Get the image
            image_scaled = sample_batch[0][10].numpy()

            # Check the range of values for this image
            logging.info(f"max value: {np.max(image_scaled)}")
            logging.info(f"min value: {np.min(image_scaled)}")

            # Step 2: Data Transformation
            logging.info("Starting data transformation.")
            # Perform data transformation
            (
                train_dataset_scaled,
                val_dataset_scaled,
                _,
            ) = self.data_transformation_service.initiate_data_transformation(
                train_tf_dataset, val_tf_dataset, test_tf_dataset
            )
            logging.info(f"Data normalized successfully")
            # Get one batch of data
            sample_batch = list(train_dataset_scaled.take(1))[0]

            # Get the image
            image_scaled = sample_batch[0][10].numpy()

            # Check the range of values for this image
            logging.info(f"max value: {np.max(image_scaled)}")
            logging.info(f"min value: {np.min(image_scaled)}")

            # Step 3: Model Training and Selection
            logging.info("Starting model training and selection.")

            # MODEL_TYPE = "mobile"
            # MODEL_TYPE = "densenet"
            # MODEL_TYPE = "efficientnet"
            MODEL_TYPE = "resnet"
            # model, model_file_name = create_model(MODEL_TYPE)
            # logging.info(model.summary())
            # model, model_file_name = create_model_from_config(MODEL_TYPE)
            model, model_file_name = create_model_from_config(MODEL_TYPE)
            # Model summary
            logging.info(model.summary())

            # Configure the training dataset
            train_dataset_final = (
                train_dataset_scaled.cache()
                .shuffle(self.config.SHUFFLE_BUFFER_SIZE)
                .batch(batch_size=self.config.BATCH_SIZE)
                .prefetch(  # Use constant from Config
                    self.config.PREFETCH_BUFFER_SIZE
                )  # Use constant from Config
            )

            # Configure the validation dataset
            validation_dataset_final = (
                val_dataset_scaled.cache()
                .batch(batch_size=self.config.BATCH_SIZE)
                .prefetch(self.config.PREFETCH_BUFFER_SIZE)
            )  # Use constant from Config

            history, metadata = self.model_training_service.train_and_validate(
                model=model,
                model_type=MODEL_TYPE,
                model_file_name=model_file_name,
                train_dataset=train_dataset_final,
                val_dataset=validation_dataset_final,
            )

            logging.info("Pipeline completed successfully.")
            results = {"history": history, "metadata": metadata}

            return results

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys) from e
