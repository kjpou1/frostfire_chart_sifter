import asyncio

from src.exception import CustomException
from src.logger_manager import LoggerManager
from src.models.command_line_args import CommandLineArgs
from src.pipeline.train_pipeline import TrainPipeline
from src.services.data_ingestion_service import DataIngestionService

logging = LoggerManager.get_logger(__name__)


class Host:
    """
    Host class to manage the execution of the main application.

    Handles initialization with command-line arguments and execution
    of the specified subcommands (e.g., ingest, train).
    """

    def __init__(self, args: CommandLineArgs):
        """
        Initialize the Host class with command-line arguments.

        Parameters:
        args (CommandLineArgs): Command-line arguments passed to the script.
        """
        self.args = args
        logging.info("Host initialized with arguments: %s", self.args)

    def run(self):
        """
        Synchronously run the asynchronous run_async method.

        This is a blocking call that wraps the asynchronous method.
        """
        return asyncio.run(self.run_async())

    async def run_async(self):
        """
        Main asynchronous method to execute the host functionality.

        Determines the action based on the provided subcommand.
        """
        try:
            logging.info("Starting host operations.")

            if self.args.command == "ingest":
                logging.info("Executing data ingestion workflow.")
                await self.run_ingestion()
            elif self.args.command == "train":
                logging.info("Executing training workflow.")
                await self.run_training()
            else:
                logging.error("No valid subcommand provided.")
                raise ValueError(
                    "Please specify a valid subcommand: 'ingest' or 'train'."
                )

        except CustomException as e:
            logging.error("A custom error occurred during host operations: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise
        finally:
            logging.info("Shutting down host gracefully.")

    async def run_ingestion(self):
        """
        Execute the data ingestion workflow.
        """
        data_ingestion_service = DataIngestionService()
        await data_ingestion_service.initiate_data_ingestion()

    async def run_training(self):
        """
        Execute the model training workflow.
        """
        train_pipeline = TrainPipeline()
        await train_pipeline.run_pipeline()
