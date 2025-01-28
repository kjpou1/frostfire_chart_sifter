from dataclasses import dataclass
from typing import Optional


@dataclass
class CommandLineArgs:
    command: str
    config: str
    debug: bool
    model_type: Optional[str] = None
