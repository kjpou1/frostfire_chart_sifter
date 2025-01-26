import json
import os
import random
from src.config.config import Config
from src.models.dataset_split_config import DatasetSplitConfig
from src.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class DatasetSplitterService:
    def __init__(self):
        """
        Initialize the service with base directories.

        Args:
            raw_dir (str): Directory where raw JSONL files are stored.
            processed_dir (str): Directory where processed files will be saved.
        """
        self.dataset_split_config = Config()
        self.raw_dir = self.dataset_split_config.RAW_DATA_DIR
        self.processed_dir = self.dataset_split_config.PROCESSED_DATA_DIR

    def split_dataset(
        self,
        jsonl_path,
        prefix,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    ):
        """
        Split a JSONL dataset into train, validation, and test sets for a specific prefix.

        Args:
            jsonl_path (str): Path to the input JSONL file.
            prefix (str): Prefix for the dataset (e.g., 'crypto', 'stock').
            train_ratio (float): Proportion of the dataset for training.
            val_ratio (float): Proportion for validation.
            test_ratio (float): Proportion for testing.
            seed (int): Random seed for reproducibility.
        """
        try:
            # Ensure output directories exist
            train_dir = os.path.join(self.processed_dir, "train")
            val_dir = os.path.join(self.processed_dir, "val")
            test_dir = os.path.join(self.processed_dir, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Read all records from the JSONL file
            with open(jsonl_path, "r") as f:
                records = [json.loads(line) for line in f]

            # Shuffle the records
            random.seed(seed)
            random.shuffle(records)

            # Compute split indices
            total = len(records)
            train_idx = int(total * train_ratio)
            val_idx = train_idx + int(total * val_ratio)

            # Split the data
            train_records = records[:train_idx]
            val_records = records[train_idx:val_idx]
            test_records = records[val_idx:]

            # Update the image paths to absolute paths
            for split_name, split_records in zip(
                ["train", "val", "test"], [train_records, val_records, test_records]
            ):
                split_dir = os.path.join(self.raw_dir, "train", "images")
                for record in split_records:
                    record["image"] = os.path.join(
                        os.path.abspath(split_dir), os.path.basename(record["image"])
                    )

            # Save each split to separate JSONL files
            splits = {
                "train": train_records,
                "val": val_records,
                "test": test_records,
            }
            for split_name, split_records in splits.items():
                output_file = os.path.join(
                    self.processed_dir, split_name, f"{prefix}.jsonl"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    for record in split_records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                logging.info(f"Saved {len(split_records)} records to {output_file}")
        except Exception as e:
            logging.error(f"Error processing dataset {prefix}: {e}")
            raise

    def split_datasets(self, datasets):
        """
        Process all datasets by splitting each into train, val, and test sets.

        Args:
            datasets (list): List of datasets with 'dataset_name' and 'prefix'.
        """
        for dataset in datasets:
            try:
                dataset_name = dataset["dataset_name"]
                prefix = dataset["prefix"]
                raw_file_path = os.path.join(
                    self.raw_dir, "train", f"{prefix}_train.jsonl"
                )

                if not os.path.exists(raw_file_path):
                    logging.warning(f"Raw file not found: {raw_file_path}. Skipping...")
                    continue

                logging.info(
                    f"Processing dataset: {dataset_name} with prefix: {prefix}..."
                )
                self.split_dataset(jsonl_path=raw_file_path, prefix=prefix)
            except Exception as e:
                logging.error(
                    f"Failed to process dataset {dataset['dataset_name']}: {e}"
                )

    def initiate_split(self):
        # Ensure the directory for saving artifacts exists
        os.makedirs(os.path.dirname(self.processed_dir), exist_ok=True)

        # Define datasets to download
        datasets_to_split = [
            {"dataset_name": "StephanAkkerman/crypto-charts", "prefix": "crypto"},
            {"dataset_name": "StephanAkkerman/stock-charts", "prefix": "stock"},
            {"dataset_name": "StephanAkkerman/fintwit-images", "prefix": "fintwit"},
        ]

        self.split_datasets(datasets_to_split)


# Example Usage
if __name__ == "__main__":
    raw_dir = "./data/raw/"
    processed_dir = "./data/processed/"

    datasets_to_process = [
        {"dataset_name": "StephanAkkerman/crypto-charts", "prefix": "crypto"},
        {"dataset_name": "StephanAkkerman/stock-charts", "prefix": "stock"},
        {"dataset_name": "StephanAkkerman/fintwit-images", "prefix": "fintwit"},
    ]

    splitter = DatasetSplitterService(raw_dir=raw_dir, processed_dir=processed_dir)
    splitter.split_datasets(datasets_to_process)
