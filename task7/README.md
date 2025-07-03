# Breast Cancer Classification using Support Vector Machines (SVM)

This task demonstrates the use of **Support Vector Machines (SVM)** for binary classification using the **Breast Cancer dataset**.

The goal is to classify tumors as **malignant** or **benign** using both **linear and non-linear (RBF) SVMs**, evaluate their performance, and visualize the decision boundary in 2D using PCA.

## Objectives

The implementation includes:
- SVM training with **linear** and **RBF** kernels.
- **Hyperparameter tuning** using `GridSearchCV`.
- **Cross-validation** to assess model stability.
- **Dimensionality reduction** using PCA.
- **Decision boundary visualization** for model interpretation.

## Dataset

- **Source**: [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- **Target Variable**: `diagnosis` (M = Malignant, B = Benign)

## Tools & Libraries Used

  **Python**
- `pandas`, `numpy` – data manipulation
- `matplotlib`, `seaborn` – visualization
- `scikit-learn` – ML models, PCA, GridSearchCV, evaluation


##  Results
The result will be:
- Print accuracy scores
- Show classification reports
- Display a 2D decision boundary plot.

The screenshots are added to 'result'.