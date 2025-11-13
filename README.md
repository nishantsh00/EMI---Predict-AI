# 💰 EMIPredict AI – Intelligent EMI Risk Assessment Platform

**EMIPredict AI** is a machine learning–powered FinTech application built using **XGBoost** and **Streamlit**, designed to predict:
1. **EMI Eligibility** — whether a customer qualifies for EMI financing.
2. **Maximum EMI Amount** — the maximum safe EMI the customer can afford.

This project combines predictive modeling, explainable AI, and an interactive user interface for smarter financial decision-making.

---

## 🚀 Features

- 🧠 **Dual-Model Prediction**
  - **Classification Model** → Predicts EMI eligibility (0 = Not Eligible, 1 = High Risk, 2 = Eligible)
  - **Regression Model** → Predicts maximum EMI amount
- ⚙️ **XGBoost Models** for high-accuracy predictions
- 💻 **Streamlit Web App** for real-time, interactive results
- 🔠 **Manual One-Hot Encoding** for model-aligned feature mapping (44 total features)
- 🔒 No external encoders or scalers required
- ☁️ Ready for **Streamlit Cloud Deployment**

---

## 🧩 Project Structure

📦 EMIPredictAI/
│
├── app.py # Streamlit web application
├── requirements.txt # Project dependencies
├── xgboost_classification_model.pkl # Trained EMI eligibility model
├── xgboost_regression_model.pkl # Trained EMI amount model
└── README.md # Project documentation

## 🧮 Input Features

### 🔢 Numeric Inputs (17)

| Feature                | Description                    |
| ---------------------- | ------------------------------ |
| age                    | Applicant’s age                |
| monthly_salary         | Monthly income                 |
| years_of_employment    | Years of work experience       |
| monthly_rent           | House rent per month           |
| family_size            | Total family members           |
| dependents             | Number of dependents           |
| school_fees            | Monthly school expenses        |
| college_fees           | Monthly college expenses       |
| travel_expenses        | Monthly travel costs           |
| groceries_utilities    | Monthly groceries & utilities  |
| other_monthly_expenses | Miscellaneous monthly expenses |
| current_emi_amount     | Existing EMI commitments       |
| credit_score           | Credit score (300–850)         |
| bank_balance           | Current savings                |
| emergency_fund         | Emergency reserves             |
| requested_amount       | Requested loan amount          |
| requested_tenure       | Tenure in months               |

---

### 🔠 Categorical Inputs (27 One-Hot Encoded Columns)

- **Gender:** Male, Female  
- **Marital Status:** Single, Married  
- **Education:** High School, Graduate, Post Graduate, Professional, Others  
- **Employment Type:** Private, Government, Self-employed, Others  
- **Company Type:** Small, Medium, Large, Multinational  
- **House Type:** Rented, Own, Family  
- **Existing Loans:** Yes, No  
- **EMI Scenario:** E-commerce, Education, Home Appliances, Personal Loan, Vehicle
