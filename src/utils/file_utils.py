import json
import os
import sys

import dill  # Serialization library for Python objects
import tensorflow as tf

from src.config.config import Config
from src.exception import CustomException
from src.logger_manager import LoggerManager

# Initialize logger
logging = LoggerManager.get_logger(__name__)


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to the specified file path using the `dill` library.

    Args:
        file_path (str): The file path where the object will be saved.
        obj (object): The Python object to serialize and save.

    Raises:
        CustomException: If an error occurs during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save object to {file_path}: {e}")
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Load a Python object from a file using `dill`.

    Args:
        file_path (str): Path to the file.

    Returns:
        object: The loaded Python object.

    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
            logging.info(f"Object loaded successfully from {file_path}")
            return obj
    except Exception as e:
        logging.error(f"Failed to load object from {file_path}: {e}")
        raise CustomException(e, sys) from e


def save_json(file_path: str, obj: object) -> None:
    """
    Saves a Python object as a JSON file.

    Args:
        file_path (str): The path where the JSON file will be saved.
        obj (object): The Python object to be serialized to JSON.

    Raises:
        CustomException: If the object cannot be serialized or if there's an issue writing to the file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
        logging.info(f"JSON saved successfully to {file_path}")
    except TypeError as e:
        logging.error(f"Unable to serialize object to JSON. Error: {e}")
        raise CustomException(f"Unable to serialize object to JSON. Error: {e}") from e
    except OSError as e:
        logging.error(f"Failed to write JSON file to {file_path}. Error: {e}")
        raise CustomException(
            f"Failed to write JSON file to {file_path}. Error: {e}"
        ) from e


def load_metadata(file_path: str) -> dict:
    """
    Load metadata from a JSON file.

    Args:
        file_path (str): Path to the metadata file.

    Returns:
        dict: Loaded metadata.

    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        with open(file_path, "r") as f:
            metadata = json.load(f)
            logging.info(f"Metadata loaded successfully from {file_path}")
            return metadata
    except Exception as e:
        logging.error(f"Failed to load metadata from {file_path}: {e}")
        raise CustomException(f"Failed to load metadata from {file_path}. Error: {e}")


def save_metadata(file_path: str, meta_data: dict):
    """
    Save metadata to a JSON file.

    Args:
        file_path (str): Path to save the metadata.
        meta_data (dict): Metadata to save.
    """
    try:
        save_json(file_path, meta_data)
        logging.info(f"Metadata saved successfully to {file_path}")
    except CustomException as e:
        logging.error(f"Failed to save metadata: {e}")
        raise


def parse_jsonl(jsonl_path: str):
    """
    Parse a JSONL file and yield image paths and labels.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Yields:
        tuple: Image path and label.
    """
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                yield item["image"], item["label"]
        logging.info(f"JSONL file parsed successfully: {jsonl_path}")
    except Exception as e:
        logging.error(f"Error parsing JSONL file {jsonl_path}: {e}")
        raise CustomException(e, sys) from e


def load_datasets_from_directory(
    data_dir: str, batch_size: int = None, img_size: tuple = None
) -> tf.data.Dataset:
    """
    Create a combined tf.data.Dataset from JSONL files in a directory.

    Args:
        data_dir (str): Directory containing JSONL files for train, val, or test splits.
        batch_size (int, optional): Batch size for the dataset. Defaults to Config.BATCH_SIZE.
        img_size (tuple, optional): Target size for images (height, width). Defaults to Config.IMG_SIZE.

    Returns:
        tf.data.Dataset: Combined TensorFlow dataset.
    """
    try:
        # Load defaults from Config if not provided
        config = Config()
        batch_size = batch_size or config.BATCH_SIZE
        img_size = img_size or config.IMG_SIZE

        jsonl_files = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith(".jsonl")
        ]

        def load_single_jsonl(jsonl_path: str) -> tf.data.Dataset:
            def generator():
                with open(jsonl_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        yield item["image"], item["label"]

            output_signature = (
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )

            return tf.data.Dataset.from_generator(
                generator, output_signature=output_signature
            )

        combined_dataset = None
        for jsonl_path in jsonl_files:
            single_dataset = load_single_jsonl(jsonl_path)
            combined_dataset = (
                single_dataset
                if combined_dataset is None
                else combined_dataset.concatenate(single_dataset)
            )

        def preprocess(image_path: str, label: int) -> tuple:
            try:
                image = tf.io.read_file(image_path)
                image = tf.image.decode_png(image, channels=3)
                image = tf.image.resize(image, img_size)
                return image, label
            except tf.errors.NotFoundError:
                logging.error(f"File not found: {image_path.numpy().decode('utf-8')}")
                return None, None

        combined_dataset = (
            combined_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=1000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        logging.info(f"Dataset loaded successfully from directory: {data_dir}")
        return combined_dataset

    except Exception as e:
        logging.error(f"Failed to load dataset from directory {data_dir}: {e}")
        raise CustomException(e, sys) from e


def save_training_artifacts(history, metadata, run_id):
    """
    Save training artifacts, including the training history and metadata, to the history directory.

    Args:
        history: The history object returned by `model.fit`.
        metadata: A dictionary containing metadata about the training run.
        run_id: Unique identifier for this training run.

    Raises:
        CustomException: If there is an error during saving.
    """
    try:
        config = Config()
        history_dir = config.HISTORY_DIR
        history_file = os.path.join(history_dir, f"history_{run_id}.json")
        metadata_file = os.path.join(history_dir, f"metadata_{run_id}.json")

        save_json(history_file, history.history)
        logging.info(f"Training history saved to {history_file}")

        save_json(metadata_file, metadata)
        logging.info(f"Training metadata saved to {metadata_file}")
    except Exception as e:
        logging.error(f"Failed to save training artifacts: {str(e)}")
        raise CustomException(f"Failed to save training artifacts: {str(e)}")
