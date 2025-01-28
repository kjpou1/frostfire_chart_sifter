import os
import sys

import numpy as np
from tensorflow.keras.utils import img_to_array, load_img

from src.exception import CustomException
from src.logger_manager import LoggerManager

# from keras.utils import img_to_array, load_img


logging = LoggerManager.get_logger(__name__)


class ImagePreprocessingService:
    """
    Service to preprocess image data for prediction.
    """

    def __init__(self, target_size=(224, 224)):
        """
        Initialize the ImagePreprocessingService.

        Args:
            target_size (tuple): Target size for resizing images (height, width).
        """
        self.target_size = target_size
        logging.info(
            f"ImagePreprocessingService initialized with target size {self.target_size}."
        )

    def preprocess_images(self, file_paths):
        """
        Preprocess images from file paths into NumPy arrays.

        Args:
            file_paths (list): List of image file paths.

        Returns:
            np.ndarray: Preprocessed images as NumPy arrays.
        """
        try:
            logging.info("Starting image preprocessing.")
            images = []
            for path in file_paths:
                logging.info(f"Processing image: {path}")
                # Load and resize the image
                img = load_img(path, target_size=self.target_size)
                images.append(img)

            logging.info("Image preprocessing completed successfully.")
            return np.array(images)

        except Exception as e:
            logging.error(f"Error during image preprocessing: {e}")
            raise CustomException(e, sys) from e
            logging.error(f"Error during image preprocessing: {e}")
            raise CustomException(e, sys) from e
