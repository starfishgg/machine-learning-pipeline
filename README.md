# Machine Learning Pipeline Framework

A modular machine learning experimentation framework built in Python using scikit-learn.

The goal is to provide a reusable architecture for training, evaluating, and comparing machine learning models across different datasets with minimal code changes.

Instead of creating separate one-off scripts for each dataset or model, the framework separates datasets, models, pipelines, evaluation, and experiment results into reusable components.

##
NOTE: The Fraud Detection Dataset must be downloaded separately as I feel it is too large for me to upload to github personally. You can download it yourself from: 

https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

It is ~200MB zipped, and ~500MB raw. Running it though this pipeline project currently takes around 10-15 minutes (it contains around 6,000,000 records). The Random Forest Classifier model does get some incredible results though (see below).

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
  - Precision
  - Recall
  - F1 score
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

### Financial Fraud Detection

**Problem type:** Classification

Trying to predict which financial transactions are fraudulent based on previous data trends. Using the dataset at: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset

Be aware that this can take a long time to run (upwards of 10-15 minutes) as it is processing ~6,000,000 records. The performance of the Random Forest Classifier on this dataset is great, given the features we engineered.

**Models:**

- Logistic Regression
- Random Forest Classifier

**Analysis:**

- Accuracy
- Confusion matrix
- Precision
- Accuracy
- F1
- Classification error analysis
- Feature analysis
- Model prediction agreement


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

```
machine-learning-pipeline/
│
├── data/
│
├── datasets/
│   ├── dataset.py
│   ├── fraud.py
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

```
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

```
==================================================
FRAUD DETECTION
==================================================

MODEL COMPARISON
----------------

LOGISTIC_REGRESSION_CLASSIFIER
Accuracy: 99.93%
Precision: 91.18%
Recall: 48.52%
F1 Score: 63.34%

CONFUSION MATRIX
================

                     Predicted
                Negative   Positive
Actual Negative   1270828          76
Actual Positive    834         786


ERROR ANALYSIS
==============

 True Negatives: 1270828
False Positives: 76
False Negatives: 834
 True Positives: 786

The model correctly classified 1271614 of 1272524 samples.


RANDOM_FOREST_CLASSIFIER
Accuracy: 100.00%
Precision: 100.00%
Recall: 99.75%
F1 Score: 99.88%

CONFUSION MATRIX
================

                     Predicted
                Negative   Positive
Actual Negative   1270904           0
Actual Positive      4        1616


ERROR ANALYSIS
==============

 True Negatives: 1270904
False Positives: 0
False Negatives: 4
 True Positives: 1616

The model correctly classified 1272520 of 1272524 samples.


BEST MODEL
----------
RANDOM_FOREST_CLASSIFIER

FEATURE ANALYSIS
----------------

LOGISTIC_REGRESSION_CLASSIFIER
type_PAYMENT: -5.7937
type_CASH_OUT: -0.7619
type_TRANSFER: 0.1160
type_DEBIT: -0.1036
type_CASH_IN: -0.0111
step: 0.0054
origin_balance_error: 0.0000
origin_balance_change: 0.0000
newbalanceOrig: -0.0000
oldbalanceOrg: 0.0000
destination_balance_change: -0.0000
amount: -0.0000
newbalanceDest: -0.0000
oldbalanceDest: 0.0000
destination_balance_error: 0.0000

RANDOM_FOREST_CLASSIFIER
newbalanceOrig: 0.2831
origin_balance_change: 0.1965
origin_balance_error: 0.1713
newbalanceDest: 0.0672
destination_balance_error: 0.0562
destination_balance_change: 0.0498
amount: 0.0416
step: 0.0328
type_TRANSFER: 0.0305
oldbalanceOrg: 0.0293
oldbalanceDest: 0.0266
type_CASH_OUT: 0.0108
type_PAYMENT: 0.0022
type_CASH_IN: 0.0021
type_DEBIT: 0.0001

MODEL PREDICTION DIFFERENCE
---------------------------

Same predictions: 1271618/1272524
Different predictions: 906/1272524

==================================================
EXPERIMENT COMPLETE
==================================================

Datasets processed : 1
Models evaluated   : 2
```

---

## Author Notes

Developed as a personal project to explore machine learning engineering, reusable software architecture, and experiment management using Python.
