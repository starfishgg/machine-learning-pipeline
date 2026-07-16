# Machine Learning Pipeline Framework

A modular machine learning experimentation framework built in Python using scikit-learn.

The goal of this project is to provide a reusable architecture for training, evaluating, and comparing machine learning models across different datasets with minimal code changes.

Instead of creating one-off scripts for each dataset, the framework separates datasets, models, evaluation, and reporting into reusable components.

---

## Features

* Object-oriented architecture
* Dataset abstraction
* Model abstraction
* Machine learning pipelines
* Classification and regression support
* Model comparison
* Feature importance / coefficient analysis
* Confusion matrix reporting
* Regression metrics (MAE, RMSE, R²)
* Factory pattern for model creation
* Factory pattern for evaluator creation
* Easily extensible for new datasets and models

---

## Current Datasets

### Titanic Survival Prediction

Classification problem.

Models:

* Logistic Regression
* Random Forest Classifier

Outputs:

* Accuracy
* Confusion Matrix
* Error Analysis
* Feature Analysis
* Model Agreement

---

### California Housing Prices

Regression problem.

Models:

* Linear Regression
* Random Forest Regressor

Outputs:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Feature Analysis
* Prediction Difference Analysis

---

## Project Structure

```text
machine-learning-pipeline/
│
├── data/
│
├── datasets/
│   ├── dataset.py
│   ├── titanic.py
│   └── california_housing.py
│
├── models/
│   ├── classification/
│   ├── regression/
│   ├── model.py
│   ├── model_factory.py
│   ├── model_type.py
│   └── problem_type.py
│
├── pipeline/
│   ├── machine_learning_pipeline.py
│   └── experiment_runner.py
│
├── evaluation/
│   ├── classification_evaluator.py
│   ├── regression_evaluator.py
│   └── evaluator_factory.py
│
├── results/
│
├── utils/
│
├── main.py
└── README.md
```

---

## Design

The framework follows a modular pipeline:

```text
Dataset
    │
    ▼
Preprocessing
    │
    ▼
Model Training
    │
    ▼
Prediction
    │
    ▼
Evaluation
    │
    ▼
Feature Analysis
    │
    ▼
Model Comparison
```

Core design principles include:

* Abstract base classes
* Factory pattern
* Separation of concerns
* Composition over inheritance where appropriate
* Extensible architecture

---

## Technologies

* Python 3.12
* pandas
* NumPy
* scikit-learn

---

## Future Improvements

* Wine Quality dataset
* Breast Cancer dataset
* Product Sales Forecasting
* K-Means clustering
* Cross-validation
* Hyperparameter tuning
* Model persistence
* Automatic report generation
* SHAP feature explanations

---

## Example Output

```text
MODEL COMPARISON

LINEAR_REGRESSION
MAE : 0.533
RMSE: 0.746
R²  : 0.576

RANDOM_FOREST_REGRESSOR
MAE : 0.328
RMSE: 0.506
R²  : 0.805

BEST MODEL
RandomForestRegressor
```

---

## Author Notes

Developed as a personal project to explore machine learning engineering, reusable software architecture, and experiment management using Python.
