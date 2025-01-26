import sys

import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.models.data_transformation_config import DataTransformationConfig

logging = LoggerManager.get_logger(__name__)


class DataTransformationService:
    """
    Handles data transformation, including preprocessing, feature scaling,
    and normalizing TensorFlow datasets.
    """

    def __init__(self):
        """
        Initializes the DataTransformationService class with the configuration.
        """
        logging.info("Initializing DataTransformationService...")
        self.data_transformation_config = DataTransformationConfig()
        self.rescale_layer = tf.keras.layers.Rescaling(
            scale=1.0 / 255
        )  # Normalization layer
        logging.info("DataTransformationService initialized successfully.")

    def normalize_dataset(self, dataset):
        """
        Normalize a tf.data.Dataset using a Rescaling layer.

        Args:
            dataset (tf.data.Dataset): The tf.data.Dataset to normalize.

        Returns:
            tf.data.Dataset: A normalized tf.data.Dataset with pixel values scaled to [0, 1].
        """
        try:
            logging.info("Normalizing dataset using Rescaling layer...")
            normalized_dataset = dataset.map(
                lambda image, label: (self.rescale_layer(image), label),
                num_parallel_calls=tf.data.AUTOTUNE,
            ).prefetch(tf.data.AUTOTUNE)
            logging.info("Dataset normalization completed successfully.")
            return normalized_dataset
        except Exception as e:
            logging.error("Error occurred during dataset normalization: %s", str(e))
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_dataset, val_dataset, test_dataset):
        """
        Accepts raw TensorFlow datasets, applies normalization, and outputs normalized datasets.

        Args:
            train_dataset (tf.data.Dataset): Raw training dataset.
            val_dataset (tf.data.Dataset): Raw validation dataset.
            test_dataset (tf.data.Dataset): Raw testing dataset.

        Returns:
            tuple: Normalized TensorFlow datasets for training, validation, and testing.
        """
        try:
            logging.info("Starting data transformation process...")

            # Normalize datasets
            logging.info("Normalizing training dataset...")
            train_dataset_normalized = self.normalize_dataset(train_dataset)

            logging.info("Normalizing validation dataset...")
            val_dataset_normalized = self.normalize_dataset(val_dataset)

            logging.info("Normalizing testing dataset...")
            test_dataset_normalized = self.normalize_dataset(test_dataset)

            logging.info("Data transformation process completed successfully.")
            return (
                train_dataset_normalized,
                val_dataset_normalized,
                test_dataset_normalized,
            )

        except Exception as e:
            logging.error("Error occurred during data transformation: %s", str(e))
            raise CustomException(e, sys) from e
