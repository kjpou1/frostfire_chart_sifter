import base64
import os
import sys
from io import BytesIO

import numpy as np
from PIL import Image
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

    def parse_and_preprocess_images(self, base64_images):
        """
        Decode Base64 images, resize, normalize, and convert to NumPy array.

        Args:
            base64_images (list): List of Base64-encoded images.

        Returns:
            np.ndarray: Preprocessed images as NumPy arrays.
        """
        try:
            logging.info("Starting Base64 image parsing and preprocessing.")
            images = []
            placeholder_image = np.zeros((*self.target_size, 3), dtype=np.float32)

            for index, base64_image in enumerate(base64_images):
                try:
                    # Decode Base64 string into image data
                    image_data = base64.b64decode(base64_image)
                    img = Image.open(BytesIO(image_data)).convert("RGB")

                    # Resize the image
                    img = img.resize(self.target_size)

                    images.append(img)
                    logging.info(f"Successfully processed Base64 image #{index + 1}.")
                except Exception as e:
                    logging.error(f"Error processing Base64 image #{index + 1}: {e}")
                    # Append None for failed images (can handle later if needed)
                    images.append(placeholder_image)

            # Remove any failed images (None entries)
            images = [img for img in images if img is not None]

            if len(images) == 0:
                raise ValueError("No valid images were processed.")

            # Stack into a single NumPy array
            processed_images = np.array(images)
            logging.info(
                f"Base64 image parsing and preprocessing completed successfully. Processed {len(images)} images."
            )
            return processed_images

        except Exception as e:
            logging.error(f"Error during Base64 image preprocessing: {e}")
            raise CustomException(e, sys) from e
