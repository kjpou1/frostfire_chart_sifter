import logging
import os
import uuid
from datetime import datetime

import tensorflow as tf

from src.config.config import Config
from src.utils.file_utils import save_training_artifacts
from src.utils.report_utils import save_training_report


class ModelTrainingService:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)

    def train_and_validate(
        self,
        model,
        model_type,
        model_file_name,
        train_dataset,
        val_dataset,
        additional_callbacks=None,
    ):
        try:
            # Configure checkpoint saving
            checkpoint_path = os.path.join(self.config.MODEL_DIR, model_file_name)
            save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )

            # Combine callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                save_checkpoint,
            ]
            if additional_callbacks:
                callbacks.extend(additional_callbacks)

            # Train the model
            self.logger.info("Starting model training.")
            EPOCHS = self.config.EPOCHS
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1,
            )

            # Generate metadata and save artifacts
            run_id = (
                datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
            )
            metadata = {
                "model_type": model_type,
                "model_file_name": model_file_name,
                "epochs": EPOCHS,
                "batch_size": self.config.BATCH_SIZE,
                "checkpoint_path": checkpoint_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            save_training_artifacts(history, metadata, model_type, run_id)

            # Step 3: Generate Reports
            save_training_report(history, model_type, run_id)
            logging.info("Pipeline reports generated successfully.")

            self.logger.info("Model training completed successfully.")
            return history, metadata
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise
