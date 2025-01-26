import os
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    Defines the file path for saving the preprocessor object.
    """
