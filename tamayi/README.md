# Customer Churn Prediction in the Telecommunications Industry

This project is part of the **Applied Artificial Intelligence (AAI)** program at the **University of San Diego (USD)**. The project analyzes customer behavior and predicts customer churn using machine learning to support customer retention initiatives in the telecommunications industry.

**Course:** AAI-510 - Machine Learning: Fundamentals and Applications </br>
**Group:** Group 6 </br>
**Project Status:** In Progress </br>

---

## Project Overview

Customer churn is one of the biggest challenges faced by telecommunications companies. Acquiring new customers is usually far more expensive than retaining existing ones. By identifying customers who are likely to leave, organizations can proactively apply retention strategies and reduce revenue loss.

This project develops predictive machine learning models that classify whether a customer is likely to churn, based on demographic information, contract type, payment method, and billing characteristics.

The work follows an end-to-end machine learning workflow:

* Data understanding
* Exploratory Data Analysis (EDA)
* Data preparation and feature engineering
* Feature selection
* Model development and tuning
* Model evaluation
* Deployment discussion
* Business insights and retention recommendations

---

## Project Objectives

* Analyze customer demographics and account patterns.
* Identify factors that contribute to customer attrition.
* Build predictive models for churn classification.
* Compare multiple machine learning algorithms.
* Evaluate model performance using classification metrics.
* Generate actionable insights to support retention programs.
* Demonstrate an end-to-end AI solution suitable for business use.

---

## Contributors

* **Ved Prakash Dwivedi**
* **Jagdish Mane**
* **Tamayi Mlanda**

---

## Faculty Advisor

* **Prof. Azka Azka**
  University of San Diego, Applied Artificial Intelligence Program

---

## Methods Used

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* Supervised Machine Learning
* Model Evaluation
* Business Analytics
* Classification Modeling

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* XGBoost
* Jupyter Notebook
* Git and GitHub

---

## Dataset Description

**Dataset:** Telco Customer Churn Data (synthetic)

**Source:** Kaggle, by Dhrubang Talukdar

**Dataset URL:** https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data

**License:** Apache 2.0

This is a fully synthetic dataset of 100,000 customer records, generated in Python with seeded randomness. It contains no real customer information, which avoids privacy concerns while preserving realistic telecom billing, contract, and demographic patterns. The target variable indicates whether a customer discontinued the telecom service.

### Target Variable

* **Churn**
  * Yes: the customer left the company
  * No: the customer stayed with the company

The Kaggle page describes the churn rate as roughly 20 percent, but the file actually used here is about 33 percent Yes (33,144 of 100,000 records). We report the measured rate throughout the analysis and treat the dataset as a mildly imbalanced classification problem.

### Data Dictionary

The dataset has nine columns.

| Column | Description | Type | Values or range |
| --- | --- | --- | --- |
| CustomerID | Unique customer identifier, dropped before modeling | int | 1 to 100000 |
| Age | Customer age in years | int | 18 to 80 |
| Gender | Customer gender | string | Female, Male, Other |
| Tenure | Months the customer has stayed with the company | int | 1 to 72 |
| MonthlyCharges | Monthly bill in USD | float | roughly 10 to 150 |
| TotalCharges | Total billed over tenure (Tenure times MonthlyCharges plus small noise) | float | roughly 16 to 10600 |
| Contract | Contract type | string | Month-to-month, One year, Two year |
| PaymentMethod | Preferred payment method | string | Bank transfer, Credit card, Electronic check, Mailed check |
| Churn | Target variable | string | Yes, No |

### Documented Churn Drivers

The dataset author documents the synthetic churn logic, which we use as hypotheses to validate during EDA and feature engineering. Churn probability is higher when:

* tenure is short (under 12 months),
* monthly charges are high (over 100),
* the contract is month-to-month.

---

## Machine Learning Tasks

### 1. Data Understanding and Exploratory Data Analysis

Objectives:

* Understand customer demographics and account characteristics.
* Explore the churn distribution.
* Identify relationships between features and churn.
* Detect missing values, duplicates, and anomalies.
* Visualize important patterns and validate the documented churn drivers.

### 2. Data Preparation and Feature Engineering

Activities include:

* Handling missing values and validating TotalCharges.
* Treating outliers.
* Encoding categorical variables (Gender, Contract, PaymentMethod).
* Scaling numerical variables.
* Deriving features (e.g. tenure buckets, average charge per month, senior-age flag).
* Addressing the mild class imbalance where appropriate.

### 3. Feature Selection

* Select features using correlation analysis and model-based importance.
* Justify the retained feature set based on the data analysis.

### 4. Classification Modeling

Models considered:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* Support Vector Machine

Top models are tuned, and a voting ensemble is evaluated.

### 5. Model Evaluation

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report

Special emphasis is placed on Recall and ROC-AUC, because missing a likely churner has a higher business cost than a false alarm.

What each metric means for this project:

| Metric | What it means for this project |
| --- | --- |
| Accuracy | Of all customers in the test split, the share the model labels correctly as churn or stay. It is a useful headline number, but with about 33 percent churn it can look high while still missing churners, so we do not rely on it alone for retention decisions. |
| Precision | Of the customers the model flags as likely to churn, the share who actually churn. High precision means a retention campaign wastes little budget on customers who were never going to leave, keeping the cost of discounts and outreach focused on genuine churn risk. |
| Recall | Of the customers who actually churn, the share the model successfully flags. This is our priority metric, because a missed churner is lost revenue. We accept some false alarms to catch more leavers, and we tune the decision threshold below 0.5 to raise recall. |
| F1-Score | The harmonic mean of precision and recall, summarising both in one number. It rewards a model that catches churners (recall) without flooding the retention team with false alarms (precision). We use it to compare models when we want balanced performance rather than favouring one error type. |
| ROC-AUC | The probability that the model scores a random churner higher than a random non-churner, across all thresholds. At about 0.80 our model ranks churn risk well above chance. Because it is threshold-independent, it is our main tool for comparing models before we pick an operating threshold. |
| Confusion Matrix | A two-by-two table of predictions versus reality: churners caught, churners missed (false negatives), false alarms (false positives), and stayers correctly cleared. It makes the costly false negatives visible, letting us see how many likely churners would slip through at a given decision threshold. |
| Classification Report | A scikit-learn summary printing precision, recall, and F1 for both the churn and stay classes, plus support counts. It reports performance per class rather than overall, confirming the model does not earn its scores by simply favouring the larger stay class and neglecting churners. |

### 6. Deployment (Discussion)

Following the assignment guidance, deployment is described rather than fully built. The discussion covers the deployment type (batch scoring versus real-time API), latency and cost considerations, model serialization, and where the model could be hosted, along with monitoring for data drift.

#### Working example application

To make the discussion concrete, the `deployment/` folder contains a small, runnable example that serves the trained model. It is split into two parts:

* **API (`deployment/api/`).** A FastAPI service that loads `best_model.pkl` and `preprocessor.pkl`, rebuilds the engineered features exactly as the feature-engineering notebook, applies the saved preprocessor, then scores. It exposes `/predict` (one customer), `/predict/batch` (a CSV of many customers), `/schema` (the seven input fields), and `/health`.
* **Client (`deployment/client/`).** A Streamlit application that calls the API over HTTP. A user can score a single customer through a guided form, upload a CSV for batch scoring and download the results, and adjust the decision threshold (defaulting to the recall-favoring 0.40 from the evaluation). The form is built from the API's `/schema`, so the allowed values live in one place.

An optional, fully self-contained extra: when an OpenAI-compatible LLM is configured (e.g. DeepSeek, Qwen), the Streamlit app offers a plain-text path where the user describes the customer and the assistant collects the seven fields conversationally before scoring. With no LLM key the manual form and batch upload still work end to end.

This example is intentionally discussion-grade rather than production-hardened: the data is synthetic, there is no authentication, and Gender (which carries no predictive signal) should be excluded from any real decisioning. See `deployment/README.md` for full run instructions.

---

## Business Impact

Accurate churn prediction lets a telecom company:

* improve customer retention,
* reduce revenue loss,
* design targeted intervention campaigns,
* optimize marketing spend,
* increase customer lifetime value,
* support data-driven decision-making.

---

## Project Structure

```text
tamayi/
│
├── data/
│   ├── synthetic_customer_churn_100k.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── Final Project SectionX-Team 6.ipynb
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
│
├── deployment/
│   ├── api/                  FastAPI scoring service
│   │   ├── main.py           endpoints: /health, /schema, /predict, /predict/batch, /chat
│   │   ├── model.py          loads artifacts, engineers features, scores
│   │   ├── schemas.py        request and response validation
│   │   └── llm.py            optional LLM-guided field collection
│   │
│   ├── client/               Streamlit example application
│   │   └── app.py            manual form, optional plain-text chat, batch CSV upload
│   │
│   └── README.md
│
├── reports/
│   └── final_report.pdf
│
├── pyproject.toml            project dependencies (notebooks + deployment)
├── uv.lock                   pinned, reproducible lockfile
├── LICENSE
└── README.md
```

The five numbered notebooks form the working pipeline. The `Final Project SectionX-Team 6.ipynb` notebook is the single report-style submission notebook that assembles every required section end to end. Replace the `X` with the team's section number before submitting.

The `deployment/` folder serves the trained model. The model artifacts are saved as `best_model.pkl` (the fitted estimator) and `preprocessor.pkl` (a single `ColumnTransformer` that scales numerics and one-hot encodes categoricals), with `metrics.json` recording every candidate model's scores.

---

## Installation and Usage

### Clone the repository

```bash
git clone https://github.com/JagdishMane/ml-telco-customer-churn-prediction.git
cd ml-telco-customer-churn-prediction/tamayi
```

### Install dependencies

This project uses [uv](https://docs.astral.sh/uv/) to manage the environment. One command creates an isolated `.venv` and installs the exact pinned versions from `uv.lock`:

```bash
uv sync
```

Then prefix any command with `uv run` to execute it inside that environment, or activate the venv directly. If you prefer plain pip, you can export a `requirements.txt` from the lockfile with `uv export --no-hashes -o requirements.txt` and `pip install -r requirements.txt` into a venv of your own.

### Launch Jupyter Notebook

```bash
uv run jupyter notebook
```

Run the notebooks in order, from `01_data_understanding.ipynb` through `05_model_evaluation.ipynb`. Notebook `03` writes the processed feature matrix to `data/processed/`, and notebook `04` saves the trained model artifacts to `models/`.

### Run the example application (Streamlit)

The deployment example has two parts that run as separate processes. Use two terminals, both started from the `tamayi/` folder (the same folder you cloned into above). `uv sync` already installed everything they need.

Terminal 1, start the API:

```bash
uv run uvicorn deployment.api.main:app --reload
```

Terminal 2, start the Streamlit client:

```bash
uv run streamlit run deployment/client/app.py
```

The API serves on `http://localhost:8000` (interactive docs at `/docs`), and the Streamlit app opens in the browser and calls it. The optional LLM path is enabled by copying `deployment/api/.env.example` to `deployment/api/.env` and setting the LLM variables. See `deployment/README.md` for details.

In Visual Studio Code you can start both the API and the Streamlit client at once with `Ctrl+Shift+B` (Run Build Task). This runs the default task defined in `.vscode/tasks.json`, which launches each service in its own terminal panel.

---

## Team Contributions

### Tamayi Mlanda

* Exploratory Data Analysis
* Modeling
* Documentation
* Streamlit example application

### Ved Prakash Dwivedi

* Feature Engineering
* Modeling
* Presentation

### Jagdish Mane

* Modeling
* Evaluation
* Deployment

---

## Future Enhancements

* Fully deploy the model as an interactive application.
* Integrate SHAP for model explainability.
* Enable batch predictions through file uploads.
* Expose the model through a REST API.
* Implement automated retraining pipelines.
* Monitor model drift and prediction quality.

---

## References

* Kaggle Telco Customer Churn Data (Dhrubang Talukdar)
* Scikit-learn documentation
* XGBoost documentation
* Applied Artificial Intelligence coursework materials

---

## License

This project is licensed under the **Apache License 2.0**. See the accompanying `LICENSE` file for the full text.

Copyright 2026 Group 6:

* Ved Prakash Dwivedi
* Jagdish Mane
* Tamayi Mlanda

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
