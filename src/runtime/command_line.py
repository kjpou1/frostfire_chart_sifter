import argparse

from src.models.command_line_args import CommandLineArgs
from src.utils.yaml_loader import load_supported_model_types

from .logging_argument_parser import LoggingArgumentParser


class CommandLine:
    @staticmethod
    def parse_arguments() -> CommandLineArgs:
        """
        Parse command-line arguments and return a CommandLineArgs object.

        Supports subcommands like 'ingest' and 'train'.
        """
        parser = LoggingArgumentParser(description="Frostfire Chart Sifter Application")

        # Create subparsers for subcommands
        subparsers = parser.add_subparsers(dest="command", help="Subcommands")

        # Subcommand: ingest
        ingest_parser = subparsers.add_parser(
            "ingest", help="Download and prepare datasets."
        )
        ingest_parser.add_argument(
            "--config",
            type=str,
            required=False,
            help="Path to the configuration file for ingestion.",
        )
        ingest_parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode during ingestion.",
        )

        # Subcommand: train
        train_parser = subparsers.add_parser(
            "train", help="Train the model using the configured pipeline."
        )
        train_parser.add_argument(
            "--config",
            type=str,
            required=False,
            default="config/model_config.yaml",
            help="Path to the configuration file for training.",
        )
        train_parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode during training.",
        )
        train_parser.add_argument(
            "--model_type",
            type=str,
            required=True,
            help="Specify the type of model to train (e.g., 'mobile', 'custom-1', 'resnet').",
        )

        # Parse the arguments
        args = parser.parse_args()

        # Check if a subcommand was provided
        if args.command is None:
            parser.print_help()
            exit(1)

        # Validate model type for the "train" command
        if args.command == "train":
            supported_model_types = load_supported_model_types(args.config)
            if args.model_type not in supported_model_types:
                print(
                    f"Error: Unsupported model type '{args.model_type}'. "
                    f"Supported types are: [{', '.join(supported_model_types)}]."
                )
                exit(1)

        # Return a CommandLineArgs object with parsed values
        return CommandLineArgs(
            command=args.command,
            config=args.config,
            debug=args.debug,
            model_type=getattr(args, "model_type", None),
        )
