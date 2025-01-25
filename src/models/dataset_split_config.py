from dataclasses import dataclass
import os


@dataclass
class DatasetSplitConfig:
    """
    Configuration for hugging face.
    Defines the file path for saving the preprocessor object.
    """

    raw_data_dir: str = os.path.join("artifacts", "raw")
    processed_data_dir: str = os.path.join("artifacts", "processed")
