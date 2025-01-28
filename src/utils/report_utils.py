import os

from matplotlib import pyplot as plt

from src.config.config import Config
from src.logger_manager import LoggerManager
from src.utils.file_utils import save_json

# Initialize logger
logging = LoggerManager.get_logger(__name__)


def save_training_report(history, model_type, run_id):
    """
    Generates and saves training plots and a summary report.

    Args:
        history: The history object returned by `model.fit`.
        model_type: Name of the model type (e.g., mobile, efficientnet).
        run_id: Unique identifier for the training run.
    """
    try:
        config = Config()

        # Create directories for reports and plots
        report_dir = os.path.join(config.REPORTS_DIR, model_type)
        plot_dir = os.path.join(report_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Extract metrics from history
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(acc))

        # Save the accuracy plot
        plt.figure()
        plt.plot(epochs, acc, "r", label="Training accuracy")
        plt.plot(epochs, val_acc, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend(loc=0)
        accuracy_plot_file = os.path.join(plot_dir, f"accuracy_plot_{run_id}.png")
        plt.savefig(accuracy_plot_file)
        plt.close()
        logging.info(f"Accuracy plot saved to {accuracy_plot_file}")

        # Save the loss plot
        plt.figure()
        plt.plot(epochs, loss, "r", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend(loc=0)
        loss_plot_file = os.path.join(plot_dir, f"loss_plot_{run_id}.png")
        plt.savefig(loss_plot_file)
        plt.close()
        logging.info(f"Loss plot saved to {loss_plot_file}")

        # Generate a summary report
        final_summary = {
            "run_id": run_id,
            "model_type": model_type,
            "final_training_accuracy": acc[-1],
            "final_validation_accuracy": val_acc[-1],
            "final_training_loss": loss[-1],
            "final_validation_loss": val_loss[-1],
            "epochs": len(acc),
            "best_epoch": val_acc.index(max(val_acc)) + 1,
            "best_validation_accuracy": max(val_acc),
        }
        summary_file = os.path.join(report_dir, f"training_summary_{run_id}.json")
        save_json(summary_file, final_summary)
        logging.info(f"Training summary saved to {summary_file}")

    except Exception as e:
        logging.error(f"Failed to generate training report: {str(e)}")
        raise
