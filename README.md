# Customer Churn Prediction in the Telecommunications Industry

This project is part of the **Applied Artificial Intelligence (AAI)** program at the **University of San Diego (USD)**. The project aims to analyze customer behavior and predict customer churn using machine learning techniques to support customer retention initiatives in the telecommunications industry.

**Course:** AAI-510 - Machine learning: Fundamentals and Applications </br>
**Group:** Group 6 </br>
**Project Status:** In Progress </br>

---

## 📌 Project Overview

Customer churn represents one of the biggest challenges faced by telecommunications companies. Acquiring new customers is often significantly more expensive than retaining existing ones. By identifying customers who are likely to leave, organizations can proactively implement retention strategies and minimize revenue loss.

This project develops predictive machine learning models to classify whether a customer is likely to churn based on demographic information, service subscriptions, billing details, and account characteristics.

The project follows an end-to-end machine learning workflow, including:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model development
* Model evaluation
* Business insights generation
* Recommendations for customer retention strategies

---

## 🎯 Project Objectives

* Analyze customer demographics and service usage patterns.
* Identify factors contributing to customer attrition.
* Develop predictive models for churn classification.
* Compare multiple machine learning algorithms.
* Evaluate model performance using classification metrics.
* Generate actionable insights to support retention programs.
* Demonstrate an end-to-end AI solution suitable for business applications.

---

## 👥 Contributors

* **Ved Prakash Dwivedi**
* **Jagdish Mane**
* **Tamayi Mlanda**

---

## 🎓 Faculty Advisor

* **Prof. Azka Azka**
  University of San Diego – Applied Artificial Intelligence Program

---

## 🧠 Methods Used

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* Supervised Machine Learning
* Model Evaluation
* Business Analytics
* Classification Modeling

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* XGBoost
* Jupyter Notebook
* Git & GitHub

---

## 📊 Dataset Description

**Dataset:** Telco Customer Churn Data

**Source:** Kaggle

Dataset URL:

https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data

The dataset contains approximately **100,000 customer records** and includes demographic information, account details, subscribed services, billing information, and churn labels. The target variable indicates whether a customer has discontinued the telecom service.

### Target Variable

* **Churn**

  * Yes: Customer left the company
  * No: Customer remained with the company

### Features Included

Examples of variables include:

* Gender
* Senior Citizen Status
* Tenure
* Contract Type
* Internet Service Type
* Payment Method
* Monthly Charges
* Total Charges
* Multiple Lines
* Tech Support
* Online Security
* Streaming Services
* Partner Status
* Dependents

---

## 🤖 Machine Learning Tasks

### 1. Exploratory Data Analysis

Objectives:

* Understand customer demographics.
* Explore churn distributions.
* Identify relationships between features and churn.
* Detect missing values and anomalies.
* Visualize important patterns.

---

### 2. Feature Engineering

Activities include:

* Handling missing values.
* Encoding categorical variables.
* Scaling numerical variables.
* Feature selection.
* Addressing class imbalance (if applicable).

---

### 3. Classification Modeling

Models considered include:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* Support Vector Machine
* Neural Networks (if applicable)

---

### 4. Model Evaluation

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report

Special emphasis will be placed on Recall and ROC-AUC due to the business impact of missing potential churners.

---
## 5. Model Deployment

To demonstrate a production-oriented workflow, the final churn prediction model will be deployed as an interactive web application.

### Deployment Objectives

* Provide a user-friendly interface for churn prediction.
* Allow business users to input customer information.
* Generate real-time churn predictions.
* Display churn probabilities to support retention decisions.

### Deployment Technology

* **Framework:** Streamlit
* **Model Serialization:** Pickle
* **Environment Management:** requirements.txt
* **Containerization (Optional):** Docker

### Deployment Workflow

Customer Inputs
↓
Data Preprocessing
↓
Feature Engineering
↓
Trained Model
↓
Churn Probability Prediction
↓
Business Recommendation

The deployed application enables stakeholders without technical expertise to leverage machine learning insights for proactive customer retention.


## 📈 Business Impact

Accurate churn prediction enables telecom companies to:

* Improve customer retention.
* Reduce revenue loss.
* Design targeted intervention campaigns.
* Optimize marketing expenditures.
* Increase customer lifetime value.
* Support data-driven decision-making.

---

## ⚙️ Installation & Usage

### Clone the Repository

```bash
git clone https://github.com/JagdishMane/ml-telco-customer-churn-prediction.git

cd ml-telco-customer-churn-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## 📁 Project Structure

```text
ml-telco-customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── metrics.json
│
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── config/
│   └── config.yaml
│
├── reports/
│   ├── figures/
│   ├── presentation/
│   └── final_report.pdf
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_prediction.py
│
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```
---
## Running the Deployment Application

After training and saving the final model:

1. Navigate to the deployment folder:

```bash
cd deployment
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit application:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal (typically `http://localhost:8501`).

Users can enter customer attributes and obtain churn predictions instantly.

---

## 👨‍💻 Team Contributions

Based on the project status update:

### Tamayi Mlanda

* Exploratory Data Analysis
* Modeling
* Documentation

### Ved Prakash Dwivedi

* Feature Engineering
* Modeling
* Presentation

### Jagdish Mane

* Modeling
* Evaluation
* Deployment

---

## 🚀 Future Enhancements

Potential future improvements include:

- Deploy the application to Streamlit Community Cloud.
- Integrate SHAP for model explainability.
- Enable batch predictions through file uploads.
- Develop REST APIs using FastAPI.
- Implement automated retraining pipelines.
- Monitor model drift and prediction quality.

---

## 📚 References

* Kaggle Telco Customer Churn Dataset
* Scikit-learn Documentation
* XGBoost Documentation
* Applied Artificial Intelligence Coursework Materials

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

You are free to:

* ✅ Use this project for personal and commercial purposes
* ✅ Modify and distribute the source code
* ✅ Incorporate the code into larger projects
* ✅ Create derivative works

Under the terms of the Apache License 2.0, users must:

* Retain the original copyright notice.
* Include a copy of the Apache 2.0 license in redistributions.
* State significant changes made to the original work.

For more details, see the accompanying `LICENSE` file in this repository.

Copyright © 2026 Group 6:

* Ved Prakash Dwivedi
* Jagdish Mane
* Tamayi Mlanda

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.

You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

