import os
import sys

import numpy as np

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.services.data_ingestion_service import DataIngestionService
from src.services.data_transformation_service import DataTransformationService
from src.services.dataset_splitter_service import DatasetSplitterService
from src.services.hugging_face_service import HuggingFaceService
from src.services.model_selection_service import ModelSelectionService
from src.utils.file_utils import save_object
from src.utils.ml_utils import create_model

logging = LoggerManager.get_logger(__name__)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_service = DataIngestionService()
        self.data_transformation_service = DataTransformationService()
        self.model_selection_service = ModelSelectionService()
        self.config = Config()

    def run_pipeline(self):
        try:

            # Step 1: Data Ingestion
            logging.info("Starting data ingestion.")
            # train_path, test_path = (
            #     self.data_ingestion_service.initiate_data_ingestion()
            # )
            train_tf_dataset, val_tf_dataset, test_tf_dataset = (
                self.data_ingestion_service.initiate_data_ingestion()
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
                test_dataset_scaled,
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
            model, model_file_name = create_model("mobile_large")

            # Model summary
            logging.info(model.summary())

            # Load Config
            config = Config()

            # Configure the training dataset
            train_dataset_final = (
                train_dataset_scaled.cache()
                .shuffle(self.config.SHUFFLE_BUFFER_SIZE)  # Use constant from Config
                .prefetch(self.config.PREFETCH_BUFFER_SIZE)  # Use constant from Config
            )

            # Configure the validation dataset
            validation_dataset_final = val_dataset_scaled.cache().prefetch(
                self.config.PREFETCH_BUFFER_SIZE
            )  # Use constant from Config

            # Configure the test dataset
            test_dataset_final = test_dataset_scaled.cache().prefetch(
                self.config.PREFETCH_BUFFER_SIZE
            )  # Use constant from Config

            results = {}
            # results = self.model_selection_service.initiate_model_trainer(
            #     train_arr, test_arr
            # )
            # logging.info(f"Model training and selection complete. Results: {results}")

            # # Step 4: Save Artifacts
            # logging.info("Saving artifacts.")
            # # Save the best model
            # logging.info("Saving the best model.")
            # save_object(
            #     file_path=self.model_selection_service.model_trainer_config.trained_model_file_path,
            #     obj=results["best_model"],
            # )

            # save_object(preprocessor_path, preprocessor_path)
            # save_object(
            #     self.model_selection_service.model_trainer_config.trained_model_file_path,
            #     os.path.join("artifacts", "model.pkl"),
            # )
            logging.info("Artifacts saved successfully.")
            return results

        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise CustomException(e, sys) from e
