import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager

import tensorflow as tf
import uvicorn
from fastapi import Depends, FastAPI, Request

from src.exception import CustomException
from src.models.image_payload import ImagePayload
from src.pipeline.predict_pipeline import PredictPipeline
from src.services.ImagePreprocessingService import ImagePreprocessingService

logger = logging.getLogger(__name__)


def get_predict_pipeline(app: FastAPI):
    """
    Retrieve the loaded model from FastAPI's application state.
    """
    if app.state.predict_pipeline:
        return app.state.predict_pipeline
    else:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown logic.
    """
    # Startup logic
    logger.info("Starting application...")
    app.state.predict_pipeline = PredictPipeline("densenet", "binary")
    logger.info("Application started successfully.")

    # Yield control to the application
    yield

    # Shutdown logic (if needed)
    logger.info("Shutting down resources...")


class Host:
    def __init__(self, args: None):
        """
        Initialize the Host class for Frostfire Stock Analysis AI Hub.
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", 8000))

        # Initialize FastAPI
        self.app = FastAPI(
            title="Frostfire Chart Sifter API", version="1.0", lifespan=lifespan
        )
        self.setup_routes()

    def setup_routes(self):
        """
        Set up FastAPI routes.
        """

        @self.app.post("/sift_images/")
        async def detect_charts(
            request: Request,
            predict_pipeline: tf.keras.Model = Depends(
                lambda: get_predict_pipeline(self.app)
            ),
        ):
            """
            Detect if Base64-encoded images are charts using DenseNet.

            Parameters:
            - request (Request): Raw request data.

            Returns:
            - dict: Structured response with code, code_text, message, and data.
            """
            try:
                # Parse and validate the payload using ImagePayload
                body = await request.json()
                payload = ImagePayload(body)
                base64_images = payload.base64_images
            except ValueError as e:
                self.logger.error("Invalid request payload: %s", e)
                return {
                    "code": 400,
                    "code_text": "error",
                    "message": str(e),
                    "data": None,
                }

            results = []
            preprocessing = ImagePreprocessingService()
            features = preprocessing.parse_and_preprocess_images(base64_images)

            # Perform prediction using the saved files
            predictions = predict_pipeline.predict(features)
            # Parse predictions into results format
            results = [
                {"index": idx, "is_chart": pred["label"]}
                for idx, pred in enumerate(predictions)
            ]

            # Structured response
            return {
                "code": 0,
                "code_text": "ok",
                "message": "Processed successfully.",
                "data": results,
            }

        @self.app.get("/health")
        async def health_check():
            """
            Health check endpoint to verify that the model and LLM are loaded properly.
            """
            try:
                # Check if the model is loaded
                if (
                    not hasattr(self.app.state, "predict_pipeline")
                    or not self.app.state.predict_pipeline
                ):
                    raise ValueError("Chart detection model is not initialized.")

                # Structured response
                return {
                    "code": 0,
                    "code_text": "ok",
                    "message": "All services are running.",
                    "data": {"sift_images": "loaded"},
                }

            except Exception as e:
                self.logger.error("Health check failed: %s", e)
                return {
                    "code": 500,
                    "code_text": "error",
                    "message": str(e),
                    "data": {"sift_images": "not loaded"},
                }

    def run(self):
        """
        Asynchronous method to start both MQTT and FastAPI server concurrently.
        """
        self.logger.info("Starting host process.")
        fastapi_task = None  # Initialize fastapi_task to None

        try:
            # # Start the heartbeat task
            # heartbeat_task = asyncio.create_task(self.mqtt_service.heartbeat())

            # Start FastAPI server as a task
            fastapi_task = asyncio.run(self.start_fastapi())

            # # Keep the process running until interrupted
            # while True:
            #     await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info("Stopping host process.")
        finally:
            if fastapi_task:  # Check if fastapi_task is initialized
                fastapi_task.cancel()
                fastapi_task
            # await self.mqtt_service.shutdown()

    async def start_fastapi(self):
        """
        Run the FastAPI server asynchronously.
        """
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    try:
        args = None
        instance = Host(args)

        # Run the async main function with the parsed arguments
        instance.run()
    except CustomException as e:
        logging.error("Critical error: %s. Application cannot start.", e)
