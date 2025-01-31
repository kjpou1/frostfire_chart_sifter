# Frostfire_Chart_Sifter

**Frostfire_Chart_Sifter** is a state-of-the-art machine learning project that classifies images into **charts** or **non-charts** using advanced Convolutional Neural Networks (CNNs). Designed for financial engineers, traders, and developers, it automates chart detection and integrates seamlessly into data workflows.

## Purpose

The **Frostfire_Chart_Sifter** was developed to allow users to **train machine learning models** for classifying images as **charts** or **non-charts**. Instead of being a fixed detection tool, this project provides the infrastructure to build, fine-tune, and deploy models that can be integrated into larger financial and analytical workflows.

A key application of the system is in automating the process of collecting and labeling stock charts for future analysis. Users can train models on their own datasets, optimizing them for specific use cases and improving detection accuracy over time. Once trained, these models can be deployed into automated workflows for seamless chart identification.

By providing a flexible and scalable framework for model training, **Frostfire_Chart_Sifter** empowers financial professionals, traders, and researchers to develop their own **custom classification models**, making it an essential component of a data-driven stock analysis pipeline.


---

## Table of Contents
- [Frostfire\_Chart\_Sifter](#frostfire_chart_sifter)
  - [Purpose](#purpose)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Datasets](#datasets)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
      - [Subcommand: `ingest`](#subcommand-ingest)
      - [Subcommand: `train`](#subcommand-train)
      - [Argument Validation:](#argument-validation)
    - [Debug Mode](#debug-mode)
    - [Notes](#notes)
  - [Configuration](#configuration)
  - [Reports and Analysis](#reports-and-analysis)
      - [**Purpose**](#purpose-1)
    - [**Directory Structure**](#directory-structure)
    - [**Artifacts Generated**](#artifacts-generated)
    - [**Example Outputs**](#example-outputs)
      - [**Accuracy Plot**](#accuracy-plot)
      - [**Loss Plot**](#loss-plot)
    - [**How to Use These Reports**](#how-to-use-these-reports)
  - [Model Configuration File Documentation](#model-configuration-file-documentation)
    - [Structure of the Configuration](#structure-of-the-configuration)
      - [**General Format**](#general-format)
      - [Model Configuration Documentation](#model-configuration-documentation)
    - [**Metrics Configuration**](#metrics-configuration)
    - [Input Example](#input-example)
    - [Output Example](#output-example)
    - [**Key Sections**](#key-sections)
      - [1. **`mobile`**: Example for MobileNetV3Small](#1-mobile-example-for-mobilenetv3small)
      - [2. **`custom-1`**: Custom Model with Sequential Layers](#2-custom-1-custom-model-with-sequential-layers)
      - [3. **`custom-mobile`**: MobileNetV3Small with a Learning Rate Schedule](#3-custom-mobile-mobilenetv3small-with-a-learning-rate-schedule)
    - [Key Considerations](#key-considerations)
    - [How to Use the Model Configuration](#how-to-use-the-model-configuration)
    - [Adding New Models](#adding-new-models)
  - [Technologies Used](#technologies-used)
  - [Troubleshooting and Common Issues](#troubleshooting-and-common-issues)
    - [1. SSL Certificate Issues](#1-ssl-certificate-issues)
    - [2. Missing or Invalid Data](#2-missing-or-invalid-data)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

---

## Features
- **Binary Image Classification**: Differentiates between charts (e.g., stock or crypto price charts) and other images.
- **Modular Pipeline**: Includes well-defined pipelines for data ingestion, preprocessing, training, evaluation, and inference.
- **Custom and Pre-Trained Models**: Supports custom training and pre-trained CNN architectures for efficiency.
- **Extensive Dataset Support**: Processes diverse datasets, including financial charts and unrelated images, ensuring robustness.
- **Performance Optimization**: Utilizes caching, shuffling, and prefetching for faster training and evaluation.

## Datasets

Frostfire_Chart_Sifter leverages publicly available datasets from **Hugging Face** to facilitate model training. A huge thanks to **[Stephan Akkerman](https://huggingface.co/StephanAkkerman)** for collecting these foundational datasets:

| Dataset Name | Description | Hugging Face URL |
|-------------|-------------|------------------|
| **Crypto Charts** | A collection of cryptocurrency-related charts. Useful for training models to recognize crypto price movements. | [StephanAkkerman/crypto-charts](https://huggingface.co/datasets/StephanAkkerman/crypto-charts) |
| **Stock Charts** | A dataset containing various stock market charts for training chart detection models. | [StephanAkkerman/stock-charts](https://huggingface.co/datasets/StephanAkkerman/stock-charts) |
| **Fintwit Images** | A collection of financial images sourced from Twitter, containing a mix of charts and non-chart financial visuals. | [StephanAkkerman/fintwit-images](https://huggingface.co/datasets/StephanAkkerman/fintwit-images) |

These datasets serve as the foundation for training the models, enabling it to distinguish between charts and non-charts effectively. Users can also fine-tune models with their own datasets for enhanced performance.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kjpou1/frostfire_chart_sifter.git
   cd frostfire_chart_sifter
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure
```
Frostfire_Chart_Sifter/
│
├── config/                   # Configurations (ex. model_config.yaml)
├── artifacts/                # Generated data, models, logs, etc.
├── src/                      # Source code
│   ├── config/               # Configuration management
│   ├── pipeline/             # Training and inference pipelines
│   ├── services/             # Modular services for data ingestion, transformation, etc.
│   ├── utils/                # Utility functions for common tasks
│   └── models/               # Model definitions and configurations
├── tests/                    # Automated tests
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
└── setup.py                  # Package setup script
```

---

## Usage
1. **Prepare Data**:
   - Place raw datasets in the `artifacts/data/raw/` directory, or configure paths in `src/config/config.py`.

2. **Ingest Data**:
   Use the command-line argument `ingest` to download and preprocess datasets.
   ```bash
   python launch_host.py ingest --config path/to/ingestion_config.json --debug
   ```

3. **Train the Model**:
   Use the command-line argument `train` to execute the full training pipeline. Specify the model type using the `--model_type` flag.
   ```bash
   python launch_host.py train --config path/to/train_config.json --model_type mobile --debug
   ```

4. **Run Inference**:
   Use the REST API or web interface for predictions. Example:
   ```bash
   curl -X POST "http://127.0.0.1:8008/predict" -H "Content-Type: application/json" -d '{"image_path": "path/to/image.png"}'
   ```

---

### Command-Line Arguments
The application supports subcommands for streamlined workflows:

#### Subcommand: `ingest`
Downloads and preprocesses datasets for training and evaluation.

| Argument      | Description                                      | Required | Default         |
|---------------|--------------------------------------------------|----------|-----------------|
| `--config`    | Path to the ingestion configuration file         | No       | None            |
| `--debug`     | Enable debug mode for verbose logging            | No       | False           |

**Example**:
```bash
python launch_host.py ingest --config artifacts/config/ingestion.json --debug
```

---

#### Subcommand: `train`
Executes the training pipeline.

| Argument       | Description                                      | Required | Default         |
|----------------|--------------------------------------------------|----------|-----------------|
| `--config`     | Path to the training configuration file          | No       | None            |
| `--model_type` | Specifies the model type to use for training     | Yes      | None (must be set) |
| `--debug`      | Enable debug mode for verbose logging            | No       | False           |

**Example**:
```bash
python launch_host.py train --config artifacts/config/train.json --model_type efficientnet --debug
```

#### Argument Validation:
- If the `--model_type` is missing, the program will display an error and exit:
  ```bash
  Error: The --model_type argument is required for the train subcommand.
  ```
- The `--model_type` must match one of the models defined in the configuration file (e.g., `mobile`, `efficientnet`, `resnet`, etc.).

---

### Debug Mode
Adding the `--debug` flag enables detailed logging, which is useful for troubleshooting during development or testing.

**Example**:
```bash
python launch_host.py train --model_type resnet --debug
```

This will provide detailed logs, including dataset loading, model initialization, and training progress.

---

### Notes
- Ensure the `--model_type` corresponds to a valid model defined in the `model_config.yaml` file.
- The `--config` argument is optional. If not provided, the application will default to the configuration defined in the environment variables or `Config` class.
---

## Configuration
The configuration is managed using the `Config` class in `src/config/config.py`:
- **Default Directories**:
  - `artifacts/`: Base directory for generated data, models, and logs.
  - `artifacts/data/raw/`: Raw dataset storage.
  - `artifacts/data/processed/`: Processed dataset storage.
  - `artifacts/models/`: Trained model files.
  - `artifacts/logs/`: Logging directory.
  - `artifacts/history/`: Training history and metadata.

Modify the default paths or use environment variables for customization:
```bash
export BASE_DIR=/path/to/artifacts
```

## Reports and Analysis

#### **Purpose**
The reports generated after training provide a detailed summary of the model's performance and key metrics. These artifacts help in analyzing model training, debugging, and comparing results across different runs.

---

### **Directory Structure**
The reports are stored in a directory named after the model type (e.g., `mobile`), under the base `reports` directory. Each training run creates the following files:
```
reports/
└── <model_type>/
    ├── plots/
    │   ├── accuracy_plot_<run_id>.png
    │   ├── loss_plot_<run_id>.png
    └── training_summary_<run_id>.json
```

---

### **Artifacts Generated**
1. **Plots**:
    - **Accuracy Plot** (`accuracy_plot_<run_id>.png`):
        - Shows the training and validation accuracy over epochs.
        - **Purpose**: Visualizes how the model's accuracy improves or stabilizes during training.
    - **Loss Plot** (`loss_plot_<run_id>.png`):
        - Displays the training and validation loss over epochs.
        - **Purpose**: Helps track overfitting (if validation loss diverges from training loss).

2. **Training Summary**:
    - A JSON file (`training_summary_<run_id>.json`) containing key metrics for the training run:
      ```json
      {
          "run_id": "20250127_171226_c6b8dd01",
          "model_type": "mobile",
          "final_training_accuracy": 0.9013327360153198,
          "final_validation_accuracy": 0.8569995164871216,
          "final_training_loss": 0.26697471737861633,
          "final_validation_loss": 0.3225937783718109,
          "epochs": 28,
          "best_epoch": 25,
          "best_validation_accuracy": 0.8795785307884216
      }
      ```
    - **Fields Explained**:
      - `run_id`: Unique identifier for the training run.
      - `model_type`: The model configuration used for the run.
      - `final_training_accuracy` and `final_validation_accuracy`: Accuracy scores after the last epoch.
      - `final_training_loss` and `final_validation_loss`: Loss values after the last epoch.
      - `epochs`: Total epochs run.
      - `best_epoch`: The epoch where the highest validation accuracy was achieved.
      - `best_validation_accuracy`: The best validation accuracy during training.

---

### **Example Outputs**
#### **Accuracy Plot**
![Training and Validation Accuracy](attachment:/mnt/data/accuracy_plot_20250127_171226_c6b8dd01.png)

#### **Loss Plot**
![Training and Validation Loss](attachment:/mnt/data/loss_plot_20250127_171226_c6b8dd01.png)

---

### **How to Use These Reports**
1. **Identify Model Performance**:
    - Use the accuracy and loss plots to understand model behavior.
    - Review the `training_summary` to extract key metrics like final accuracy, best epoch, and validation accuracy trends.
2. **Compare Across Runs**:
    - Compare JSON summaries for different `run_id`s to analyze the impact of hyperparameter tuning or data changes.
3. **Debugging**:
    - A diverging validation loss in the plots could indicate overfitting, prompting regularization adjustments.
4. **Reproducibility**:
    - The `run_id` and `model_type` ensure reproducibility of the training run and its configuration.

---

## Model Configuration File Documentation

The **model configuration file** is a YAML file that defines all model-specific parameters, enabling the application to dynamically create, train, and manage models without hardcoding these details into the codebase. The default file is located at:

```
config/model_config.yaml
```

### Structure of the Configuration

The configuration file is organized into sections, each corresponding to a specific model type. Here's a breakdown of the key elements:

---

#### **General Format**
```yaml
<model_name>:                    # Unique name for the model
  base_model: <Base Model Name>  # Name of the pretrained model to use (if applicable)
  trainable: <true|false>        # Whether the base model's weights are trainable
  dense_units: <int>             # Number of units in the dense layer
  dropout_rate: <float>          # Dropout rate to prevent overfitting
  learning_rate: <float>         # Learning rate for the optimizer
  loss: <Loss Function>          # Loss function for model training
  metrics:                       # List of metrics to evaluate the model
    - <Eval Metric>              # Metric can be directly specified as a string (e.g., "accuracy")
    - name: <Metric Name>        # Metric as a dictionary for customization
      args:                      # Optional arguments for the metric
        <key>: <value>           # Additional arguments passed to the metric (e.g., `curve` for AUC)
  file_name: <File Name>         # File name for saving the model
```

#### Model Configuration Documentation

---

### **Metrics Configuration**
The `metrics` field supports both simple and advanced configurations:

1. **Simple Metric**:
   Directly specify the metric name as a string.
   ```yaml
   metrics:
     - accuracy
   ```

2. **Custom Metric with Arguments**:
   Specify a dictionary with the metric name and optional arguments.
   ```yaml
   metrics:
     - name: AUC
       args:
         curve: ROC
         name: auc
   ```

---

### Input Example
The following YAML snippet demonstrates how to define metrics in the configuration file:
```yaml
metrics:
  - accuracy
  - name: AUC
    args:
      name: auc
```

---

### Output Example
The above configuration will be transformed into a Python list of TensorFlow-compatible metrics:
```python
[
    "accuracy", 
    tf.keras.metrics.AUC(name="auc")
]
```

---

### **Key Sections**

#### 1. **`mobile`**: Example for MobileNetV3Small
```yaml
mobile:
  base_model: MobileNetV3Small
  trainable: false
  dense_units: 128
  dropout_rate: 0.3
  learning_rate: 0.001
  loss: binary_crossentropy
  metrics:
    - name: Accuracy
    - name: AUC
      args:
        curve: ROC
  file_name: mobilenetv3_classifier.keras
```
- **Base Model**: `MobileNetV3Small`
- **Dropout**: Adds regularization during training.
- **Metrics**: Includes Accuracy and Area Under the Curve (AUC) with a ROC curve.

---

#### 2. **`custom-1`**: Custom Model with Sequential Layers
```yaml
custom-1:
  layers:
    - type: Rescaling
      arguments: { scale: 0.00392156862745098 }  # Normalizes pixel values
    - type: Conv2D
      arguments: { filters: 16, kernel_size: [3, 3], activation: relu }
    - type: MaxPooling2D
      arguments: { pool_size: [2, 2] }
    - type: Conv2D
      arguments: { filters: 32, kernel_size: [3, 3], activation: relu }
    - type: MaxPooling2D
      arguments: { pool_size: [2, 2] }
    - type: Flatten
    - type: Dense
      arguments: { units: 512, activation: relu }
    - type: Dense
      arguments: { units: 1, activation: sigmoid }
  optimizer: RMSprop
  optimizer_args:
    learning_rate: 0.001
  loss: binary_crossentropy
  metrics:
    - name: Accuracy
    - name: AUC
      args:
        curve: ROC
  file_name: custom1_classifier.keras
```
- **Custom Layers**: Describes the architecture explicitly, including layer types and arguments.
- **Optimizer**: Uses RMSprop with additional arguments for learning rate.

---

#### 3. **`custom-mobile`**: MobileNetV3Small with a Learning Rate Schedule
```yaml
custom-mobile:
  base_model: MobileNetV3Small
  trainable: false
  dense_units: 128
  dropout_rate: 0.3
  learning_rate_schedule:
    type: ExponentialDecay
    arguments:
      initial_learning_rate: 0.001
      decay_steps: 10000
      decay_rate: 0.9
      staircase: true
  loss: binary_crossentropy
  metrics:
    - name: Accuracy
    - name: AUC
      args:
        curve: ROC
  file_name: custom_mobilenetv3_classifier.keras
```
- **Learning Rate Schedule**: Configures an exponential decay for learning rate.
- **Pretrained Model**: MobileNetV3Small is frozen (not trainable).

---

### Key Considerations
1. **Scalability**: New models can be added by extending the `models` section without changing the code.
2. **Custom Metrics**: Specify arguments for metrics (e.g., ROC curve) for better evaluation.
3. **Flexibility**: Layer-by-layer customization for models like `custom-1` supports fine-grained control.

---

### How to Use the Model Configuration
- **Default Location**: The application loads the configuration from `config/model_config.yaml`.
- **Custom Location**: Use the `--config` flag to specify a custom file path:
  ```bash
  python launch_host.py train --model_type mobile --config path/to/your_model_config.yaml
  ```
- **Validation**: The system validates that the specified `model_type` exists in the configuration before proceeding.

---

### Adding New Models
To add a new model, follow these steps:
1. Define the model parameters under the `models` section.
2. Ensure all necessary keys (e.g., `base_model`, `trainable`, `metrics`) are included.
3. If using custom layers, provide `type` and `arguments` for each layer.

Example:
```yaml
new_model:
  base_model: EfficientNetV2
  trainable: true
  dense_units: 256
  dropout_rate: 0.4
  learning_rate: 0.0001
  loss: categorical_crossentropy
  metrics:
    - name: Accuracy
  file_name: efficientnetv2_classifier.keras
```

---

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Model Architectures**: MobileNetV3, ResNet, and custom CNNs
- **Dataset Management**: Hugging Face `datasets` library
- **Utilities**: `numpy`, `pandas`, `scikit-learn`

---

## Troubleshooting and Common Issues
### 1. SSL Certificate Issues
**Error**:
```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```
**Solution**:
- Install certificates:
  ```bash
  pip install --upgrade certifi
  ```
- Use a local weight file if downloading fails.

### 2. Missing or Invalid Data
**Solution**:
- Ensure raw datasets are in the expected format (JSONL with `image` and `label` fields).

---

## Contributing
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request.

---

## License
This project is licensed under the Mozilla Public License Version 2.0. See `LICENSE` for details.

---

## Acknowledgements
Special thanks to the contributors and the TensorFlow and Hugging Face communities for their incredible tools and support.
