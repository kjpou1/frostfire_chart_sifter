import json
import os
from datasets import load_dataset
from PIL import Image
from src.logger_manager import LoggerManager

from src.models.hugging_face_config import HuggingFaceConfig
from src.utils.file_utils import save_metadata, load_metadata

logging = LoggerManager.get_logger(__name__)


class HuggingFaceService:
    def __init__(self):
        """
        Initialize the service with the raw data directory.
        """
        self.hugging_face_config = HuggingFaceConfig()
        self.raw_data_dir = self.hugging_face_config.raw_data_dir
        self.metadata_file_path = self.hugging_face_config.metadata_file_path

    def save_image(self, image, save_dir, prefix, image_id):
        """
        Save a PIL image to the specified directory with a unique prefix and ID.
        Removes any incorrect ICC profile in the process.
        """
        try:
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Remove ICC profile by re-creating the image without metadata
            image = image.convert("RGB")  # Convert to RGB for compatibility
            image_path = os.path.join(save_dir, f"{prefix}_image_{image_id}.png")

            # Save the image without embedding any ICC profile
            image.save(image_path, format="PNG", icc_profile=None)
            return image_path
        except Exception as e:
            logging.error(f"Error saving image {image_id}: {e}")
            raise

    def download_and_save(self, dataset_info):
        """
        Download a dataset from Hugging Face and save it to disk.
        Handles image data by saving images with the specified prefix.
        """
        dataset_name = dataset_info["dataset_name"]
        prefix = dataset_info["prefix"]

        logging.info(f"Downloading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name)

        logging.info(f"Processing dataset: {dataset_name} with prefix '{prefix}'...")
        for split, data in dataset.items():

            split_dir = os.path.join(self.raw_data_dir, split)
            image_dir = os.path.join(split_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            json_output_path = os.path.join(split_dir, f"{prefix}_{split}.jsonl")
            with open(json_output_path, "w", encoding="utf-8") as f:
                for idx, example in enumerate(data):
                    try:
                        # Handle image data
                        if "image" in example and isinstance(
                            example["image"], Image.Image
                        ):
                            image_path = self.save_image(
                                example["image"], image_dir, prefix, idx
                            )
                            # Replace image object with the saved file path
                            example["image"] = image_path

                        # Serialize the rest of the example as JSON
                        json_line = json.dumps(example, ensure_ascii=False)
                        f.write(json_line + "\n")
                    except Exception as e:
                        logging.warning(
                            f"Skipping problematic record {idx} in {split}: {e}"
                        )

            logging.info(f"Saved {split} data to {json_output_path}")

    def is_dataset_processed(self, dataset_name, prefix):
        """
        Check if the dataset split has already been processed using metadata.
        """
        metadata = load_metadata(self.metadata_file_path)
        key = f"{dataset_name}_{prefix}"
        return key in metadata

    def update_metadata(
        self,
        dataset_name,
        prefix,
    ):
        """
        Update the metadata file with the processed dataset information.
        """
        metadata = load_metadata(self.metadata_file_path)
        key = f"{dataset_name}_{prefix}"
        metadata[key] = {
            "status": "processed",
        }
        save_metadata(self.metadata_file_path, metadata)

    def process_datasets(self, datasets_to_download):
        """
        Process multiple datasets by downloading and saving them.
        """
        for dataset_info in datasets_to_download:
            dataset_name = dataset_info["dataset_name"]
            prefix = dataset_info["prefix"]

            # Check if this dataset split is already processed
            if self.is_dataset_processed(dataset_name, prefix):
                logging.info(
                    f"Dataset {dataset_name}, prefix {prefix} is already processed. Skipping."
                )
                continue

            try:
                self.download_and_save(dataset_info)
                self.update_metadata(dataset_name=dataset_name, prefix=prefix)
            except Exception as e:
                logging.error(
                    f"Failed to process dataset {dataset_info['dataset_name']}: {e}"
                )

    def initiate_download(self):
        # Ensure the directory for saving artifacts exists
        os.makedirs(os.path.dirname(self.metadata_file_path), exist_ok=True)

        # Initialize metadata
        if not os.path.exists(self.metadata_file_path):
            with open(self.metadata_file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

        # Define datasets to download
        datasets_to_download = [
            {"dataset_name": "StephanAkkerman/crypto-charts", "prefix": "crypto"},
            {"dataset_name": "StephanAkkerman/stock-charts", "prefix": "stock"},
            {"dataset_name": "StephanAkkerman/fintwit-images", "prefix": "fintwit"},
        ]

        self.process_datasets(datasets_to_download)


# if __name__ == "__main__":
#     # Define datasets to download
#     datasets_to_download = [
#         {"dataset_name": "StephanAkkerman/crypto-charts", "prefix": "crypto"},
#         {"dataset_name": "StephanAkkerman/stock-charts", "prefix": "stock"},
#         {"dataset_name": "StephanAkkerman/fintwit-images", "prefix": "fintwit"},
#     ]

#     # Initialize the service and process datasets
#     service = HuggingFaceService(raw_data_dir="../../data/raw")
#     service.process_datasets(datasets_to_download)
