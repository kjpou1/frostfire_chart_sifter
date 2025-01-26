import json
import os
import sys

import dill  # Serialization library for Python objects
import numpy as np
import pandas as pd
import tensorflow as tf

from src.exception import CustomException

IMG_SIZE = (224, 224)  # Image size (height, width)
BATCH_SIZE = 32  # Batch size


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to the specified file path using the `dill` library.

    Args:
        file_path (str): The file path where the object will be saved. If directories in the path do not exist, they will be created.
        obj (object): The Python object to serialize and save.

    Raises:
        CustomException: If an error occurs during the saving process, wraps and raises the error with additional context.

    Example:
        >>> example_obj = {"key": "value"}
        >>> save_object("artifacts/example.pkl", example_obj)
    """
    try:
        # Extract the directory path from the given file path
        dir_path = os.path.dirname(file_path)

        # Ensure the directory exists; create it if it doesn't
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and serialize the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # If an error occurs, raise a CustomException with the error details and system information
        raise CustomException(e, sys) from e


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def load_metadata(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_metadata(file_path: str, meta_data: str):
    with open(file_path, "w") as f:
        json.dump(meta_data, f, indent=4)


def parse_jsonl(jsonl_path):
    """
    Parse a JSONL file and yield image paths and labels.
    """
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            yield item["image"], item["label"]


def load_datasets_from_directory(data_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Create a combined tf.data.Dataset from JSONL files in a directory.
    Args:
        data_dir (str): Directory containing JSONL files for train, val, or test splits.
        batch_size (int): Batch size for the dataset.
        img_size (tuple): Target size for images (height, width).
    Returns:
        tf.data.Dataset: Combined TensorFlow dataset.
    """
    # Collect all JSONL files in the directory
    jsonl_files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".jsonl")
    ]

    # Helper function to load and parse one JSONL file
    def load_single_jsonl(jsonl_path):
        """
        Create a tf.data.Dataset from a single JSONL file.
        """

        def generator():
            # Parse JSONL into image paths and labels
            with open(jsonl_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    yield item["image"], item["label"]

        # Define the output signature for the generator
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),  # For the string output
            tf.TensorSpec(shape=(), dtype=tf.int32),  # For the int output
        )

        # Create a dataset for this JSONL file
        dataset = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )
        # dataset = tf.data.Dataset.from_generator(
        #     generator, output_types=(tf.string, tf.int32), output_shapes=((), ())
        # )

        return dataset

    # Combine all datasets using flat_map
    combined_dataset = None
    for jsonl_path in jsonl_files:
        single_dataset = load_single_jsonl(jsonl_path)
        combined_dataset = (
            single_dataset
            if combined_dataset is None
            else combined_dataset.concatenate(single_dataset)
        )

    # Preprocessing pipeline
    def preprocess(image_path, label):
        # Load and decode image
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
        except tf.errors.NotFoundError:
            print(f"File not found: {image_path.numpy().decode('utf-8')}")
            return None, None

        # Resize
        image = tf.image.resize(image, img_size)
        return image, label

    # Apply preprocessing, batching, and shuffling
    combined_dataset = (
        combined_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return combined_dataset
