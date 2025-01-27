# Frostfire_Chart_Sifter

**Frostfire_Chart_Sifter** is a state-of-the-art machine learning project that classifies images into **charts** or **non-charts** using advanced Convolutional Neural Networks (CNNs). Designed for financial engineers, traders, and developers, it automates chart detection and integrates seamlessly into data workflows.


# Table of Contents

- [Frostfire\_Chart\_Sifter](#frostfire_chart_sifter)
- [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Why Frostfire\_Chart\_Sifter?](#why-frostfire_chart_sifter)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Project Structure](#project-structure)
  - [Technologies Used](#technologies-used)
  - [Troubleshooting and Common Issues](#troubleshooting-and-common-issues)
    - [1. SSL Certificate Issues](#1-ssl-certificate-issues)
    - [2. Missing or Invalid Data](#2-missing-or-invalid-data)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Features
- **Binary Image Classification**: Differentiates between charts (e.g., stock or crypto price charts) and other images.
- **Modular Pipeline**: Includes well-defined pipelines for data ingestion, preprocessing, training, evaluation, and inference.
- **Custom and Pre-Trained Models**: Supports custom training and pre-trained CNN architectures for efficiency.
- **Extensive Dataset Support**: Processes diverse datasets, including financial charts and unrelated images, ensuring robustness.
- **Performance Optimization**: Utilizes caching, shuffling, and prefetching for faster training and evaluation.

## Why Frostfire_Chart_Sifter?
1. **Efficiency**: Automates manual chart identification, saving time and effort.
2. **Scalability**: Customizable architecture adapts to different datasets and models.
3. **Accuracy**: Employs cutting-edge CNN techniques to ensure high classification precision.

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

## Usage
1. **Prepare Data**:
   - Place raw datasets in the `artifacts/data/raw/` directory, or configure paths in `src/config/config.py`.
2. **Train the Model**:
   Run the training pipeline:
   ```bash
   python src/pipeline/train_pipeline.py
   ```
3. **Run Inference**:
   Use the REST API or web interface for predictions. Example:
   ```bash
   curl -X POST "http://127.0.0.1:8008/predict" -H "Content-Type: application/json" -d '{"image_path": "path/to/image.png"}'
   ```

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

---

## Project Structure
```
Frostfire_Chart_Sifter/
│
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


