import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from pydantic import ValidationError
from werkzeug.utils import secure_filename

from src.logger_manager import LoggerManager
from src.pipeline.predict_pipeline import PredictPipeline
from src.schemas.prediction_input_schema import PredictionInputSchema
from src.services.ImagePreprocessingService import ImagePreprocessingService

logging = LoggerManager.get_logger(__name__)
application = Flask(__name__)

app = application

# Configure file upload settings
UPLOAD_FOLDER = "artifacts/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

## Route for a home page


@app.route("/")
def index():
    return render_template("index.html")


# Route to serve uploaded files
@app.route("/artifacts/uploads/<path:filename>")
def serve_uploaded_file(filename):
    """
    Serve files from the uploads directory.
    """
    uploads_dir = app.config["UPLOAD_FOLDER"]
    return send_from_directory(uploads_dir, filename)


@app.route("/sift_images", methods=["GET", "POST"])
def sift_images():
    try:
        # Ensure at least one file is uploaded
        if "images" not in request.files:
            return render_template("home.html", error="No file part in the request.")

        files = request.files.getlist("images")
        if not files:
            return render_template("home.html", error="No files uploaded.")

        # Save and validate files
        file_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                file_paths.append(file_path)
            else:
                return render_template("home.html", error="Invalid file type uploaded.")

        # Preprocessing
        preprocessor = ImagePreprocessingService()
        features = preprocessor.preprocess_images(file_paths)

        # Perform prediction using the saved files
        predict_pipeline = PredictPipeline("densenet")
        predictions = predict_pipeline.predict(features)

        results = [
            {
                "file_path": f"/artifacts/uploads/{os.path.basename(file_path)}",
                "label": result["label"],
                "score": result["score"],
            }
            for file_path, result in zip(file_paths, predictions)
        ]

        # Prepare results for rendering
        return jsonify(results)

    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return render_template(
            "home.html", error="An error occurred during prediction."
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8097)
