# Machine Learning Pipeline Framework

A modular machine learning experimentation framework built in Python using scikit-learn.

The goal is to provide a reusable architecture for training, evaluating, and comparing machine learning models across different datasets with minimal code changes.

Instead of creating separate one-off scripts for each dataset or model, the framework separates datasets, models, pipelines, evaluation, and experiment results into reusable components.

---

## Features

- Object-oriented architecture
- Abstract dataset and model classes
- Reusable machine learning pipeline
- Classification and regression support
- Multiple model comparison
- Model feature analysis
  - Feature importance for Random Forest models
  - Coefficients for linear models
- Classification evaluation
  - Accuracy
  - Confusion matrix
  - Error analysis
  - Model agreement
- Regression evaluation
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² score
  - Prediction difference analysis
- Factory pattern for model creation
- Factory pattern for evaluator creation
- Easily extensible for new datasets and models

---

## Current Datasets

### Titanic Survival Prediction

**Problem type:** Classification

The Titanic dataset is used to predict whether a passenger survived based on passenger and ticket information.

**Models:**

- Logistic Regression
- Random Forest Classifier

**Analysis:**

- Accuracy
- Confusion matrix
- Classification error analysis
- Feature analysis
- Model prediction agreement

---

### California Housing

**Problem type:** Regression

The California Housing dataset is used to predict median house values from demographic and geographic features.

**Models:**

- Linear Regression
- Random Forest Regressor

**Analysis:**

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score
- Feature analysis
- Prediction difference analysis

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
|   |   ├── logistic_regression_classifier.py
|   |   └── random_forest_classifier.py
|   |
│   ├── regression/
|   |   ├── linear_regression.py
|   |   └── random_forest_regressor.py
|   |
│   ├── model_factory.py
│   ├── model.py
│   ├── model_type.py
│   └── problem_type.py
│
├── pipeline/
│   ├── pipeline.py
│   └── experiment_runner.py
│
├── evaluation/
│   ├── evaluator_factory.py
│   ├── evaluator.py
│   ├── classification_evaluator.py
│   ├── classification_metrics.py
│   ├── regression_evaluator.py
│   └── regression_metrics.py
│
├── results/
│   ├── experiment_result.py
│   ├── model_result.py
│   ├── prediction_set.py
│   ├── prediction.py
│
├── main.py
└── README.md
```

---

## Design

The framework follows a modular pipeline:

```text
ExperimentRunner
│
├── Create Model
│
├── Create MachineLearningPipeline
│      │
│      ├── Load Dataset
│      ├── Preprocess
│      ├── Validate
│      ├── Train Model
│      ├── Generate Predictions
│      └── Create PredictionSet
│
├── Create ModelResult
│
└── Repeat for each requested model
        │
        ▼
ExperimentResult
        │
        ├── Compare Models
        ├── Compare Predictions
        └── Analyse Features
        │
        ▼
EvaluatorFactory
        │
        ├── ClassificationEvaluator
        │        │
        │        ▼
        │   ClassificationMetrics
        │
        └── RegressionEvaluator
                 │
                 ▼
            RegressionMetrics
```

### Separation of Responsibilities

**Datasets**

Responsible for loading, preprocessing, validating, and describing individual datasets.

**Models**

Provide a common interface for training, prediction, and feature analysis while allowing different scikit-learn models to be used interchangeably.

**Pipeline**

Executes one complete machine learning run for a single model.

**ExperimentRunner**

Runs the same experiment using multiple models so their results can be compared fairly.

**Results**

Store and organise the outputs produced by model runs, including predictions, individual model results, and the overall experiment result.

**Evaluation**

Calculates performance metrics appropriate to the machine learning problem type.


---

## Technologies

* Python
* pandas
* NumPy
* scikit-learn

---

## Future Improvements

* Additional datasets
* Additional models
* Model persistence
* Automatic report generation
* Additional evaluation metrics
* SHAP feature explanations
* Improved visualisation of model performance

---

## Example Output

```text
==================================================
CALIFORNIA HOUSING
==================================================

MODEL COMPARISON
----------------

LINEAR_REGRESSION
MAE: 0.533
RMSE: 0.746
R²: 0.576

RANDOM_FOREST_REGRESSOR
MAE: 0.328
RMSE: 0.506
R²: 0.805

BEST MODEL
----------
RANDOM_FOREST_REGRESSOR

FEATURE ANALYSIS
----------------

LINEAR_REGRESSION
AveBedrms: 0.7831
MedInc: 0.4487

RANDOM_FOREST_REGRESSOR
MedInc: 0.5250
AveOccup: 0.1384

MODEL PREDICTION DIFFERENCE
---------------------------

Mean prediction difference: 0.383
Maximum prediction difference: 9.119

==================================================
TITANIC SURVIVAL
==================================================

MODEL COMPARISON
----------------

LOGISTIC_REGRESSION_CLASSIFIER
Accuracy: 81.01%

ERROR ANALYSIS
==============

 True Negatives: 90
False Positives: 15
False Negatives: 19
 True Positives: 55

The model correctly classified 145 of 179 samples.

RANDOM_FOREST_CLASSIFIER
Accuracy: 81.01%

ERROR ANALYSIS
==============

 True Negatives: 88
False Positives: 17
False Negatives: 17
 True Positives: 57

The model correctly classified 145 of 179 samples.


BEST MODEL
----------
Models performed equally:
- LOGISTIC_REGRESSION_CLASSIFIER
- RANDOM_FOREST_CLASSIFIER

FEATURE ANALYSIS
----------------

LOGISTIC_REGRESSION_CLASSIFIER
Sex: 2.5927
Pclass: -0.9365
SibSp: -0.2938
Embarked_S: -0.2304
Embarked_C: 0.1899

RANDOM_FOREST_CLASSIFIER
Sex: 0.2796
Fare: 0.2564
Age: 0.2519

MODEL PREDICTION DIFFERENCE
---------------------------

Same predictions: 155/179
Different predictions: 24/179

==================================================
EXPERIMENT COMPLETE
==================================================

Datasets processed : 2
Models evaluated   : 4
```

---

## Author Notes

Developed as a personal project to explore machine learning engineering, reusable software architecture, and experiment management using Python.
