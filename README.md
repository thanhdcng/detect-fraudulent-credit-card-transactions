# Detect Fraudulent Credit Card Transactions

## Project Overview

The objective of this project is to develop a robust and efficient machine learning model to detect fraudulent credit card transactions. Credit card fraud is a significant issue in the financial industry, leading to substantial financial losses for banks and consumers. By leveraging data science and machine learning techniques, this project aims to identify fraudulent transactions with high accuracy, thus minimizing the impact of fraud on the financial ecosystem.

## Objectives

### Data Collection and Preprocessing
- **Acquire a dataset** of credit card transactions, including both fraudulent and legitimate transactions.
- **Preprocess the data** to handle missing values, normalize numerical features, and encode categorical variables.

### Exploratory Data Analysis (EDA)
- Perform **EDA** to understand the distribution of features and identify patterns or anomalies in the data.
- Visualize the data to highlight differences between fraudulent and non-fraudulent transactions.

### Feature Engineering
- Create new features that may improve the model's performance, such as transaction frequency, average transaction amount, and time-based features.
- Select the most relevant features using techniques like correlation analysis and feature importance scores.

### Model Development
- Experiment with various machine learning algorithms, including Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting.
- Implement techniques to handle class imbalance, such as oversampling, undersampling, and synthetic data generation (e.g., SMOTE).

### Model Evaluation and Validation
- Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and Area Under the ROC Curve (AUC-ROC).
- Perform cross-validation to ensure the model generalizes well to unseen data.

### Model Optimization
- Fine-tune model hyperparameters using grid search or randomized search techniques.
- Implement regularization techniques to prevent overfitting.

### Documentation and Reporting
- Document the entire process, including data collection (if applicable), preprocessing steps, model development, and evaluation.
- Prepare a comprehensive report and presentation to communicate findings and insights to stakeholders.

## Project Resources

- A PC, Mac, or Linux computer with Internet access.
- The development environment (Linux | Windows) and programming language(s) to be used will depend on the existing skills of the team members and will be chosen in consultation with the Project Supervisors.

## Project structure
detect-fraudulent-credit-card-transactions/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── EDA_and_Preprocessing.ipynb
│   ├── Feature_Engineering_and_Model_Development.ipynb
│   └── Model_Evaluation_and_Optimization.ipynb
├── reports/
│   ├── figures/
│   └── final_report.pdf
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
├── README.md
└── requirements.txt

## Team
- Duc Thanh Nguyen
- Donporn Rodkrajub
- Priya

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or any preferred IDE
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn

### Installation

Clone the repository:

```bash
git clone https://github.com/thanhdcng/detect-fraudulent-credit-card-transactions.git
cd detect-fraudulent-credit-card-transactions



