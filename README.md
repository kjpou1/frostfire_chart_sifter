# Frostfire_Chart_Sifter

**Frostfire_Chart_Sifter** is a cutting-edge machine learning project designed to classify images as either **charts** or **non-charts**. Leveraging Convolutional Neural Networks (CNNs), it provides a robust solution for automated chart detection. Whether you’re a financial engineer, trader, or developer, this project offers tools to seamlessly integrate chart recognition into your workflow.

### Key Features:
- **Binary Classification**: Differentiates between chart images (e.g., stock or crypto price charts) and non-chart images.
- **Pipeline Architecture**: Includes modular pipelines for data preprocessing, model training, evaluation, and inference.
- **Custom and Pre-trained Models**: Use pre-trained models for faster deployment or train custom models for tailored use cases.
- **Diverse Dataset Support**: Compatible with various datasets, including stock market charts, crypto charts, and unrelated images for robust classification.
- **Scalability**: Designed for experimentation and easy integration into production workflows.

### Why Frostfire_Chart_Sifter?
1. **Efficiency**: Automates chart detection, saving time and reducing manual effort.
2. **Flexibility**: Customizable architecture supports multiple datasets and models.
3. **Precision**: Achieves high accuracy in distinguishing charts from non-charts using advanced CNN architectures.

### Getting Started
Follow the setup instructions to:
1. **Prepare Data**: Organize datasets for charts and non-charts.
2. **Train Models**: Use provided pipelines for training and validation.
3. **Deploy Models**: Integrate into your systems for real-time or batch predictions.

### Applications
- **Automated Data Filtering**: Identify chart images in large datasets.
- **Financial Research**: Quickly extract and organize financial charts.
- **AI-Driven Workflows**: Enhance systems requiring visual data analysis.

For more details, explore the repository documentation and examples.

---
## Table of Contents
- [Frostfire\_Chart\_Sifter](#frostfire_chart_sifter)
    - [Key Features:](#key-features)
    - [Why Frostfire\_Chart\_Sifter?](#why-frostfire_chart_sifter)
    - [Getting Started](#getting-started)
    - [Applications](#applications)
  - [Table of Contents](#table-of-contents)
  - [Notes on Requirements and Installation](#notes-on-requirements-and-installation)
  - [Explanation of Dataset Configuration](#explanation-of-dataset-configuration)
    - [How It Works](#how-it-works)
    - [Why It’s Important](#why-its-important)
  - [Roles of Key Scripts](#roles-of-key-scripts)
    - [`predict_app`](#predict_app)
    - [`launch_host`](#launch_host)
    - [`predict_rest_api`](#predict_rest_api)
    - [Running the `predict_rest_api`](#running-the-predict_rest_api)
      - [Prerequisites:](#prerequisites)
      - [Starting the REST API Server:](#starting-the-rest-api-server)
      - [Interacting with the API:](#interacting-with-the-api)
      - [Key Notes:](#key-notes)
    - [`predict_fasthtml_app`](#predict_fasthtml_app)
      - [Key Features:](#key-features-1)
    - [Running the `predict_fasthtml_app`](#running-the-predict_fasthtml_app)
      - [Prerequisites:](#prerequisites-1)
      - [Starting the FastHTML Application:](#starting-the-fasthtml-application)
      - [Interaction:](#interaction)
  - [Logging Functionality](#logging-functionality)
    - [Configurable Environment Variables](#configurable-environment-variables)
  - [Automated Test Suite Documentation](#automated-test-suite-documentation)
  - [Test Categories and Coverage](#test-categories-and-coverage)
    - [1. REST API Tests](#1-rest-api-tests)
      - [Test Cases:](#test-cases)
    - [2. Data Ingestion Tests](#2-data-ingestion-tests)
      - [Test Cases:](#test-cases-1)
    - [3. Data Transformation Tests](#3-data-transformation-tests)
      - [Test Cases:](#test-cases-2)
    - [4. Model Selection Tests](#4-model-selection-tests)
      - [Test Cases:](#test-cases-3)
    - [Execution Instructions](#execution-instructions)
    - [Troubleshooting and Common Issues](#troubleshooting-and-common-issues)
      - [**1. SSL Certificate Issues**](#1-ssl-certificate-issues)
        - [**Cause**](#cause)
        - [**Solution**](#solution)
      - [**2. Debugging and Logging**](#2-debugging-and-logging)

---
## Notes on Requirements and Installation

The `setup.py` script dynamically parses the `requirements.txt` file for dependencies, but explicitly excludes the editable installation directive (`-e .`). This is because:

- Editable installations (`-e .`) are intended for local development and are not portable across environments.
- To work on the package in editable mode, run the following command manually:

  ```bash
  pip install -e .
  ```

---

The setup you provided is a best practice for configuring TensorFlow datasets for efficient input pipelines. Below is the explanation, detailed steps, and the updated code with comments.

---

## Explanation of Dataset Configuration

**1. Caching**:
- **Purpose**: Speeds up training by storing the dataset in memory after the first read.
- **When to Use**: When the dataset fits into memory and remains unchanged throughout training.

**2. Shuffling**:
- **Purpose**: Randomizes the order of the dataset to prevent the model from learning unintended patterns.
- **Buffer Size**: Controls how much data is loaded into the buffer for shuffling. A larger size ensures better shuffling but uses more memory.

**3. Prefetching**:
- **Purpose**: Allows the dataset to fetch the next batch while the model processes the current batch, reducing I/O bottlenecks.
- **Buffer Size**: Using `tf.data.AUTOTUNE` dynamically determines the optimal buffer size based on available CPU resources.

---

### How It Works

**1. Training Dataset**
- **`cache()`**:
  - Stores the dataset in memory after the first read, improving training speed.
- **`shuffle()`**:
  - Introduces randomness to the dataset order to prevent overfitting or bias.
  - A buffer size of `1000` is a good starting point for most datasets.
- **`prefetch()`**:
  - Preloads data for the next batch, reducing latency during training.

**2. Validation Dataset**
- **`cache()`**:
  - Improves evaluation speed by caching the dataset since it doesn’t change during validation.
- **`prefetch()`**:
  - Optimizes input pipeline performance during validation.

**3. Test Dataset**
- **`cache()`**:
  - Caches the dataset in memory, speeding up evaluation.
- **`prefetch()`**:
  - Ensures the pipeline is efficient during testing.

---

### Why It’s Important
1. **Performance**:
   - Reduces I/O bottlenecks and improves training, validation, and testing speeds.
2. **Scalability**:
   - Optimizes resource usage, especially on multi-core systems.
3. **Flexibility**:
   - Easy to tweak buffer sizes based on dataset size and available memory.

---

## Roles of Key Scripts

### `predict_app`
A Flask-based web application for end-users to interact with the prediction service. It:
- Accepts user inputs via a web form.
- Validates inputs using the `PredictionInputSchema` to ensure data quality.
- Sends inputs to the `PredictPipeline` for generating predictions.
- Displays predictions or error messages back to the user in an interactive interface.

---

### `launch_host`
An entry point for running asynchronous backend tasks. It:
- Serves as a host for machine learning pipelines or other backend services.
- Configurable via command-line arguments for deployment and runtime behavior.

---

### `predict_rest_api`
A FastAPI-based RESTful API for serving predictions programmatically. It:
- Exposes endpoints to accept structured prediction requests in JSON format.
- Validates inputs with `PredictionInputSchema` for data consistency and reliability.
- Returns structured responses, including predictions, validation errors, or internal error messages.
- Designed for integration with external applications or automation pipelines.

### Running the `predict_rest_api`

The `predict_rest_api.py` provides a RESTful API interface for programmatically interacting with the prediction service. It is designed to handle JSON-based requests and respond with structured outputs, making it ideal for integration with external applications or automation pipelines.

#### Prerequisites:
1. Install all dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

#### Starting the REST API Server:
1. Run the `predict_rest_api.py` server using **Uvicorn**:
   ```bash
   uvicorn predict_rest_api:app --host 0.0.0.0 --port 8008 --reload
   ```
   - The `--reload` flag enables automatic reloading of the server whenever code changes, useful for development.
   - The server will be accessible at:
     ```plaintext
     http://127.0.0.1:8008
     ```

#### Interacting with the API:
1. Open a REST client (e.g., **Postman**, **curl**, or a custom client application).
2. Use the `/predict` endpoint to submit prediction requests.

   **Example Request**:
   ```bash
   curl -X POST "http://127.0.0.1:8008/predict" \
   -H "Content-Type: application/json" \
   -d '{
       "payload": {
           "data": {
               "gender": "male",
               "race_ethnicity": "group A",
               "parental_level_of_education": "high school",
               "lunch": "standard",
               "test_preparation_course": "none",
               "reading_score": 72.0,
               "writing_score": 74.0
           }
       }
   }'
   ```

   **Response**:
   - On success:
     ```json
     {
         "code": 0,
         "code_text": "ok",
         "message": "Processed successfully.",
         "data": {"math_score": 76.9151611328125}
     }
     ```
   - On validation error:
     ```json
     {
         "code": -1,
         "code_text": "error",
         "message": "Validation error occurred.",
         "errors": [
             {"field": "reading_score", "error": "value is not a valid float"},
             {"field": "writing_score", "error": "field required"}
         ]
     }
     ```
   - On internal server error:
     ```json
     {
         "code": -1,
         "code_text": "error",
         "message": "An internal server error occurred.",
         "errors": null
     }
     ```

3. Verify the REST API is running by visiting:
   ```plaintext
   http://127.0.0.1:8008/
   ```
   - The root endpoint returns a simple health check message:
     ```json
     {
         "message": "FastAPI Prediction Service is running"
     }
     ```

#### Key Notes:
- The REST API validates all input data before processing and returns helpful error messages in case of validation issues.
- The server is designed for production and can be scaled using tools like **Docker** or **Kubernetes**.

---

### `predict_fasthtml_app`
A FastHTML-based web application designed as a lightweight frontend interface to interact with the REST API. It:
- Provides an elegant and responsive HTML interface for input submission.
- Communicates with the FastAPI `predict_rest_api` to send user inputs and display results.
- Handles validation errors returned by the REST API and presents them in a user-friendly format.

#### Key Features:
- **User-Friendly Interface**: 
  - A responsive form to input prediction data, styled with clean HTML and CSS for ease of use.
- **Integration with REST API**:
  - Sends POST requests to the FastAPI service's `/predict` endpoint.
  - Processes responses and displays results or error messages dynamically.

---

### Running the `predict_fasthtml_app`

The `predict_fasthtml_app.py` provides a web interface for users to interact with the prediction service via the REST API.
This simple two-step process makes it easy to run the FastHTML app and interact with the machine learning service.

#### Prerequisites:
1. Ensure that all dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI REST API using the following command:
   ```bash
   uvicorn predict_rest_api:app --host 0.0.0.0 --port 8008 --reload
   ```
   This ensures the REST API is available to handle requests from the FastHTML app.

#### Starting the FastHTML Application:
1. Run the FastHTML app:
   ```bash
   python predict_fasthtml_app.py
   ```
2. Access the FastHTML interface in your browser:
   ```plaintext
   http://127.0.0.1:8009
   ```

#### Interaction:
1. Fill out the form fields (e.g., `Gender`, `Race/Ethnicity`, `Reading Score`, etc.) in the web interface.
2. Click the **Submit** button to send the data to the REST API for prediction.
3. **Response Handling**:
   - **On success**: Displays the predicted `Math Score` in a result section.
   - **On validation error**: Shows error messages returned by the REST API (e.g., missing or invalid inputs).
   - **On server error**: Displays a generic error message, such as "An error occurred."

---

## Logging Functionality

This project features an enhanced logging system powered by `LoggerManager`. Key features include:

- **Plain Text and JSON Logs**.
- **Dynamic Log Levels**.
- **Rotating File Logs**.

### Configurable Environment Variables

| Environment Variable | Default Value | Description |
|-----------------------|---------------|-------------|
| `LOG_LEVEL`          | `INFO`        | Sets the logging level. |
| `LOG_JSON`           | `false`       | Enable JSON logs with `true`. |

---


## Automated Test Suite Documentation

The project includes a comprehensive suite of tests to validate the functionality, robustness, and reliability of key components. Below is a categorized list of all tests with their purposes and expected outcomes, updated to reflect recent changes.

---

## Test Categories and Coverage

### 1. REST API Tests
**Located in**: `tests/test_rest_api.py`

#### Test Cases:
| **Test Name**                   | **Purpose**                                                                                           | **Expected Outcome**                                                                                          |
|----------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_predict_success`           | Tests the `/predict` endpoint with valid input data.                                                  | Returns a `200 OK` response with a valid prediction.                                                         |
| `test_predict_validation_error`  | Validates that the `/predict` endpoint handles missing or invalid input fields correctly.             | Returns a `400 Bad Request` response with detailed validation errors.                                         |
| `test_predict_missing_payload`   | Ensures the `/predict` endpoint handles empty or missing JSON payloads.                               | Returns a `422 Unprocessable Entity` response with an appropriate error message.                              |
| `test_root_endpoint`             | Verifies the root (`/`) endpoint functionality.                                                       | Returns a `200 OK` response with a health check message.                                                      |

---

### 2. Data Ingestion Tests
**Located in**: `tests/test_services/test_data_ingestion.py`

#### Test Cases:
| **Test Name**                       | **Purpose**                                                                                     | **Expected Outcome**                                                                                          |
|-------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_data_ingestion_with_missing_file` | Ensures the data ingestion process handles missing input files gracefully.                      | Raises a `CustomException` caused by a `FileNotFoundError`, with a message referencing the missing file.       |
| `test_data_ingestion_creates_files`  | Validates that data ingestion correctly creates train and test files from the input dataset.     | Generated train and test files exist in the specified paths and contain valid data.                           |

---

### 3. Data Transformation Tests
**Located in**: `tests/test_services/test_data_transformation.py`

#### Test Cases:
| **Test Name**                         | **Purpose**                                                                                     | **Expected Outcome**                                                                                          |
|---------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_data_transformation`            | Ensures the data transformation process works as expected with valid inputs.                    | Returns transformed train and test arrays with expected shapes and saves the preprocessor object.             |
| `test_data_transformation_missing_columns` | Validates that the transformation process handles missing required columns appropriately.       | Raises a `CustomException` indicating the missing columns.                                                    |

---

### 4. Model Selection Tests
**Located in**: `tests/test_services/test_model_selection.py`

#### Test Cases:
| **Test Name**                       | **Purpose**                                                                                     | **Expected Outcome**                                                                                          |
|-------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_initiate_model_trainer_success` | Validates the selection and evaluation of the best model from the given data.                   | Ensures a valid model report is returned, the best model is identified, and its score meets expectations.      |
| `test_no_model_meets_threshold`     | Ensures an exception is raised when no model meets the minimum performance threshold.            | Raises a `CustomException` with a message indicating no best model was found.                                 |
| `test_invalid_data_handling`        | Tests the behavior of model selection when invalid training and testing data are provided.       | Returns a model report with `None` scores for all but Linear Regression, which should score 1.0.              |

---

### Execution Instructions
1. Ensure all dependencies are installed using:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests using `pytest`:
   ```bash
   PYTHONPATH=$(pwd) pytest -v
   ```

3. To run a specific test module, use:
   ```bash
   PYTHONPATH=$(pwd) pytest -v --log-cli-level=INFO tests/<test_module_name>.py
   ```

<!-- 4. To generate a coverage report:
   ```bash
   PYTHONPATH=$(pwd) pytest --cov=src --cov-report=html
   ```
   - View the HTML report in the generated `htmlcov` directory. -->

---

### Troubleshooting and Common Issues

This section covers potential issues you may encounter while using **Frostfire_Chart_Sifter** and provides solutions to resolve them.

---

#### **1. SSL Certificate Issues**
You might encounter errors related to SSL certificate verification, such as:
```
URL fetch failure on <URL>: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed.
```

##### **Cause**
This occurs when Python cannot verify the SSL certificate of the URL being accessed, often due to missing or outdated root certificates.

##### **Solution**
1. **Install Missing Certificates**:
   - On macOS:
     ```bash
     /Applications/Python\ 3.x/Install\ Certificates.command
     ```
   - On all platforms:
     ```bash
     pip install --upgrade certifi
     ```

2. **Set the SSL Certificate Path**:
   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

3. **Disable SSL Verification (Not Recommended for Production)**:
   ```python
   import os
   os.environ['CURL_CA_BUNDLE'] = ''
   ```

4. **Manual Weight Download**:
   Download the weights file and specify its local path in your script:
   ```python
   model.load_weights("/path/to/local/weights.h5")
   ```

---

#### **2. Debugging and Logging**
Enable TensorFlow debugging for detailed insights:
```python
tf.debugging.set_log_device_placement(True)
```

---

