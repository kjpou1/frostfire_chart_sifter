from dataclasses import dataclass


@dataclass
class CommandLineArgs:
    command: str
    config: str
    debug: bool
    model_type: str = None
