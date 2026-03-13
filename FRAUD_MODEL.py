#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE


# In[2]:


df = pd.read_csv("frauddata.csv")


# In[3]:


print("Shape:", df.shape)


# In[4]:


print("\nColumns:\n", df.columns)


# In[5]:


print("\nMissing values:\n", df.isnull().sum())


# In[6]:


# VISUALIZE TRANSACTION TYPES
plt.figure(figsize=(6,4))
sns.countplot(x="type", data=df)
plt.title("Transaction Type Distribution")
plt.show()


# In[7]:


# FRAUD COUNT PLOT
plt.figure(figsize=(5,4))
sns.countplot(x="isFraud", data=df)
plt.title("Fraud vs Non-Fraud Count")
plt.show()


# In[8]:


# FEATURE ENGINEERING
df["errorOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
df["errorDest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]



# In[9]:


# DROP IRRELEVANT COLUMNS
df = df.drop("step", axis=1)
df = df.drop(["nameOrig", "nameDest",], axis=1)


# In[10]:


# ENCODE TRANSACTION TYPE
df = pd.get_dummies(df, columns=["type"], drop_first=True)


# In[11]:


# REMOVE OUTLIERS (IQR METHOD)
def remove_outliers(data):

    fraud = data[data["isFraud"] == 1]
    normal = data[data["isFraud"] == 0]

    numeric_cols = normal.select_dtypes(include=np.number).columns

    Q1 = normal[numeric_cols].quantile(0.25)
    Q3 = normal[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((normal[numeric_cols] < (Q1 - 1.5 * IQR)) |
             (normal[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

    normal_clean = normal[mask]

    return pd.concat([normal_clean, fraud])


df = remove_outliers(df)
print("\nShape after outlier removal:", df.shape)


# In[12]:


# CORRELATION HEATMAP
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[13]:


# SPLIT DATA

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)


# In[14]:


# SMOTE BALANCING
print("\nBefore SMOTE:\n", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:\n", pd.Series(y_train_smote).value_counts())


# In[15]:


# SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


# In[16]:


# TRAIN MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=150)
}

results = {}
print("\n========== MODEL RESULTS ==========")
for name, model in models.items():

    model.fit(X_train_scaled, y_train_smote)
    pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)

    results[name] = acc

    print(f"\n{name}")
    print("Accuracy:", round(acc,4))
    print("ROC-AUC:", round(roc,4))
    print(classification_report(y_test, pred))
     # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# In[17]:


# MODEL COMPARISON GRAPH
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.show()


# In[18]:


best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBEST MODEL:", best_model_name)

best_model.fit(X_train_scaled, y_train_smote)


# =========================
# SAVE MODEL
# =========================
joblib.dump(best_model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")

