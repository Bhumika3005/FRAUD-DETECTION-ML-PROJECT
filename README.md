# Fraud Detection Machine Learning Project

A machine learning-based web application designed to detect fraudulent financial transactions in real time. The project combines data preprocessing, domain-specific feature engineering, class imbalance handling, and trained classification models with an interactive Flask web interface.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation Instructions](#installation-instructions)
- [How to Run the Application](#how-to-run-the-application)
- [Future Improvements](#future-improvements)

---

## Project Overview

Financial fraud detection is a critical challenge in digital payments. Fraudulent transactions typically make up a very small percentage of all transactions, making them difficult to identify using standard rule-based checks.

This project implements an end-to-end machine learning solution that evaluates key transaction attributes—such as transaction amounts, sender/receiver balances, and transfer types—to predict whether an incoming transaction is legitimate or fraudulent. The trained model is deployed using Flask, providing an intuitive web interface for real-time verification.

---

## Features

- **Real-Time Prediction**: Instantly analyzes transaction details submitted through a clean web form.
- **Confidence Scoring**: Displays prediction probability percentages alongside the classification result (**Legitimate Transaction ✅** or **Fraud Transaction ❌**).
- **Automated Feature Engineering**: Calculates account balance discrepancies (`errorOrig` and `errorDest`) directly from submitted inputs.
- **One-Hot Categorical Encoding**: Supports standard banking transaction types (`CASH_OUT`, `TRANSFER`, `PAYMENT`, `DEBIT`).
- **Feature Scaling**: Normalizes inputs using a pre-fitted `StandardScaler` before generating predictions.
- **Class Imbalance Handling**: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) during training to handle the small number of fraud cases.
- **Clean Web Interface**: A responsive and user-friendly interface built with HTML5 and CSS3.

---

## Technologies Used

- **Programming Language**: Python 3.8+
- **Web Framework**: Flask
- **Machine Learning**: Scikit-learn, Imbalanced-learn (SMOTE)
- **Data Analysis & Processing**: Pandas, NumPy
- **Model Serialization**: Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, Vanilla CSS3

---

## Machine Learning Models

The project evaluates and compares multiple supervised classification algorithms on historical transaction data:

1. **Decision Tree Classifier** *(Selected Best Model)*
2. **Random Forest Classifier** (`n_estimators=150`)
3. **Logistic Regression** (`max_iter=500`)

### Preprocessing & Training Highlights:
- **Feature Engineering**: Balance error discrepancies were calculated to identify irregular account balances:
  - `errorOrig = oldbalanceOrg - newbalanceOrig - amount`
  - `errorDest = newbalanceDest - oldbalanceDest - amount`
- **Outlier Filtering**: The Interquartile Range (IQR) method was applied to legitimate transactions to reduce data noise while retaining all fraud instances.
- **Data Splitting**: Stratified 80/20 train-test split (`random_state=42`) to preserve class proportions.
- **Resampling**: SMOTE was applied exclusively to the training set to address class imbalance without causing data leakage.
- **Model Persistence**: The best-performing Decision Tree model and the fitted `StandardScaler` were serialized as `fraud_model.pkl` and `scaler.pkl`.

---

## Model Performance

The models were evaluated on a test set of **768 samples**. The actual performance metrics obtained from the project notebook are summarized below:

| Model | Test Accuracy | ROC-AUC | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** *(Selected)* | **99.87%** (`0.9987`) | **0.8750** | **1.00** | **0.75** | **0.86** |
| **Random Forest** | **99.48%** (`0.9948`) | **0.8730** | **0.50** | **0.75** | **0.60** |
| **Logistic Regression** | **95.83%** (`0.9583`) | **0.9791** | **0.11** | **1.00** | **0.20** |

> **Note on Test Set Distribution**: The test dataset consisted of 768 total transactions, comprising **764 legitimate transactions** and **4 fraudulent transactions**. Because fraudulent activity represents a very small fraction of real-world financial data, overall accuracy can be naturally high. Precision, recall, and F1-score on the fraud class provide a more balanced reflection of model capability on rare positive instances.

---

## Project Structure

```text
FRAUD-DETECTION-ML-PROJECT/
│
├── app.py                  # Flask web application and prediction routes
├── FRAUD_MODEL.py          # Model training and data preparation script
├── FRAUD_MODEL.ipynb       # Jupyter Notebook with EDA, training plots, and evaluation outputs
├── fraud_model.pkl         # Serialized trained machine learning model
├── scaler.pkl              # Serialized StandardScaler for input normalization
├── frauddata.csv           # Financial transactions dataset used for training
├── requirements.txt        # Verified Python package dependencies
├── requriments.txt         # Original dependency file preserved for reference
│
├── static/
│   └── style.css           # Styling for web pages
│
└── templates/
    ├── index.html          # Transaction input form
    └── result.html         # Prediction result page
```

---

## Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Bhumika3005/FRAUD-DETECTION-ML-PROJECT.git
   cd FRAUD-DETECTION-ML-PROJECT
   ```

2. **Create and Activate a Virtual Environment** *(recommended)*:
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Application

1. **Start the Flask Server**:
   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000/
   ```

3. **Submit a Transaction**:
   - Enter the transaction amount, sender old/new balances, receiver old/new balances, and select the transaction type.
   - Click **Analyze Transaction** to view the real-time prediction and probability score.

*(Optional)* To retrain the model and regenerate serialized files:
```bash
python FRAUD_MODEL.py
```

---

## Future Improvements

- **RESTful API**: Provide API endpoints (`/predict`) to integrate prediction capabilities into existing payment gateways.
- **Database Integration**: Store submitted transactions and predictions in a database (such as PostgreSQL) for audit logs.
- **Monitoring Dashboard**: Build an administrative dashboard to visualize transaction patterns and fraud detection trends over time.
- **Deep Learning Approaches**: Explore sequence-based models (such as LSTMs) to capture sequential transaction behaviors per account.
- **Automated Alerts**: Add notification alerts (email/SMS) for transactions flagged as high-risk.
