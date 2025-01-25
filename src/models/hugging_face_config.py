from dataclasses import dataclass
import os


@dataclass
class HuggingFaceConfig:
    """
    Configuration for hugging face.
    Defines the file path for saving the preprocessor object.
    """

    metadata_file_path: str = os.path.join("artifacts", "metadata.json")
    raw_data_dir: str = os.path.join("artifacts", "raw")
