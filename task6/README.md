K-Nearest Neighbors (KNN) Classification using Iris Dataset (CSV) 

## Objective

Implement K-Nearest Neighbors (KNN) for classification using the Iris dataset loaded from a local CSV file. Tune hyperparameters, evaluate performance, and visualize decision boundaries.

 ## Tools & Libraries

    Python 
    Scikit-learn
    Pandas
    NumPy
    Matplotlib

## Workflow

1. Load Dataset:

        Read Iris dataset from Iris.csv

        Separate features (X) and target labels (y)

        Encode string labels into integers using LabelEncoder

2. Train-Test Split:

        Stratified split (80% training, 20% testing)

3. Pipeline Construction:

        Use Pipeline with StandardScaler + KNeighborsClassifier

        Hyperparameter tuning using GridSearchCV over k = 1 to 20

4. Model Evaluation:

        Best k is selected based on cross-validation

        Accuracy and Confusion Matrix computed on test data

5. Visualization:

        Confusion Matrix displayed with class labels

        Decision boundary plotted using 2 selected features (PetalLength & PetalWidth)

## Output

Automatically selected the best k value based on cross-validation
Achieved high accuracy on test data (typically near 100%)
Plotted:
Confusion Matrix
Decision Boundary for Petal features.
The screenshots of the output are added to 'result'.