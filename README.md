# Credit Scoring Model

## CodeAlpha Machine Learning Internship - Task 1

### Project Description

This project builds a machine learning model to predict whether a person is creditworthy based on credit application information.

### Dataset

The project uses the UCI Credit Approval Dataset.

- Total records: 690
- Features: 15
- Target: Credit Approval
- Approved: 1
- Rejected: 0

### Machine Learning Algorithm

Logistic Regression is used for classification.

### Data Preprocessing

The project performs:

- Missing value handling
- Numerical feature scaling
- Categorical feature encoding
- Train-test splitting

### Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Results

Accuracy: 86.96%

Precision: 85.25%

Recall: 85.25%

F1 Score: 85.25%

ROC-AUC: 96.02%

### Project Files

- `dataset/crx.data` - UCI dataset
- `train_model.py` - Model training and evaluation code
- `credit_scoring_model.pkl` - Trained machine learning model

### Tools and Technologies

- Python
- Pandas
- Scikit-learn
- Joblib
- Logistic Regression

### Conclusion

The trained model can classify credit applications as creditworthy or not creditworthy based on the available application features.