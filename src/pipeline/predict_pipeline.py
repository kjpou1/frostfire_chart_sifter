import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.services.data_transformation_service import DataTransformationService
from src.utils.yaml_loader import load_model_config

logging = LoggerManager.get_logger(__name__)


class PredictPipeline:

    def __init__(self, model_type: Optional[str] = None, output_labels: str = "chart"):
        """
        Initialize the PredictPipeline by loading the TensorFlow model and preprocessor.
        """
        try:
            self.config = Config()
            self.model_type = self.config.model_type if not model_type else model_type
            self.output_labels = output_labels
            logging.info(
                f"Using model type: {self.model_type} with output labels: {self.output_labels}"
            )

            # Load the model configuration using the utility function
            self.model_config = load_model_config()
            logging.info(f"Loaded model configuration for type: {self.model_type}")

            # Load the model from the saved path
            model_type_config = self.model_config["models"][model_type]
            model_file_name = model_type_config["file_name"]
            model_path = os.path.join(self.config.MODEL_DIR, model_file_name)
            logging.info(f"Loading TensorFlow model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            logging.info("Model loaded successfully.")

            # Initialize the DataTransformationService
            logging.info("Initializing DataTransformationService.")
            self.data_transformer = DataTransformationService()

            # Define a tf.function for predictions to reduce retracing
            @tf.function(reduce_retracing=True)
            def _predict_fn(features):
                return self.model(features)

            self._predict_fn = _predict_fn

        except Exception as e:
            logging.error(f"Error initializing PredictPipeline: {e}")
            raise CustomException(e, sys) from e

    def predict(self, features):
        """
        Predict outcomes based on the given features.

        Args:
            features (np.ndarray, list of np.ndarray, or tf.data.Dataset):
                The input features for prediction.

        Returns:
            list[dict]: List of predictions with their labels and scores.
        """
        try:
            logging.info("Starting prediction process.")

            # Normalize input based on its type
            if isinstance(features, np.ndarray):
                logging.info("Received single NumPy array. Normalizing.")
                normalized_features = self.data_transformer.normalize(features)

            elif isinstance(features, list) and all(
                isinstance(x, np.ndarray) for x in features
            ):
                logging.info("Received list of NumPy arrays. Normalizing each array.")
                normalized_features = np.array(
                    [self.data_transformer.normalize(x) for x in features]
                )

            elif isinstance(features, tf.data.Dataset):
                logging.info("Received tf.data.Dataset. Normalizing dataset.")
                normalized_features = self.data_transformer.normalize(features)

            else:
                raise ValueError(
                    "Unsupported input type for prediction. Expected a NumPy array, "
                    "list of NumPy arrays, or tf.data.Dataset."
                )

            # Perform prediction using the model
            logging.info("Making predictions with the loaded model.")
            preds = self._predict_fn(normalized_features)
            # self.model.predict(normalized_features)

            # Convert sigmoid outputs to labels based on `output_labels`
            if self.output_labels == "chart":
                results = [
                    {
                        "score": float(pred),
                        "label": "chart" if pred <= 0.5 else "non-chart",
                    }
                    for pred in preds
                ]
            elif self.output_labels == "binary":
                results = [
                    {"score": float(pred), "label": 1 if pred <= 0.5 else 0}
                    for pred in preds
                ]
            else:
                raise ValueError(f"Unsupported output_labels: {self.output_labels}")

            logging.info("Predictions completed successfully.")
            return results

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise CustomException(e, sys) from e
