import streamlit as st
import joblib
import numpy as np

# --- Load Models ---
class_model = joblib.load("xgboost_classification_model.pkl")
reg_model = joblib.load("xgboost_regression_model.pkl")

st.set_page_config(page_title="💰 EMIPredict AI", layout="wide")
st.title("💰 EMIPredict AI - Intelligent EMI Risk Assessment")

st.markdown("### Predict EMI Eligibility and Maximum EMI using AI")

# --- 1️⃣ Applicant Inputs ---
st.header("Enter Applicant Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age (years)", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional", "Others"])

with col2:
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed", "Others"])
    company_type = st.selectbox("Company Type", ["Small", "Medium", "Large", "Multinational"])
    years_of_employment = st.number_input("Years of Employment", 0.0, 50.0, 3.0, step=0.5)
    monthly_salary = st.number_input("Monthly Salary (₹)", 0.0, 1000000.0, 50000.0, step=1000.0)

with col3:
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    monthly_rent = st.number_input("Monthly Rent (₹)", 0.0, 200000.0, 10000.0, step=500.0)
    family_size = st.number_input("Family Size", 1, 15, 3)
    dependents = st.number_input("Dependents", 0, 10, 1)

with col4:
    school_fees = st.number_input("School Fees (₹)", 0.0, 100000.0, 0.0, step=500.0)
    college_fees = st.number_input("College Fees (₹)", 0.0, 200000.0, 0.0, step=1000.0)
    travel_expenses = st.number_input("Travel Expenses (₹)", 0.0, 50000.0, 2000.0, step=500.0)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0.0, 100000.0, 8000.0, step=500.0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0.0, 50000.0, 2000.0, step=500.0)

st.markdown("### Financial Status")

col5, col6 = st.columns(2)

with col5:
    existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
    current_emi_amount = st.number_input("Current EMI (₹)", 0.0, 500000.0, 0.0, step=1000.0)
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    bank_balance = st.number_input("Bank Balance (₹)", 0.0, 10000000.0, 20000.0, step=1000.0)

with col6:
    emergency_fund = st.number_input("Emergency Fund (₹)", 0.0, 10000000.0, 30000.0, step=1000.0)
    requested_amount = st.number_input("Requested Loan Amount (₹)", 0.0, 5000000.0, 50000.0, step=5000.0)
    requested_tenure = st.number_input("Requested Tenure (months)", 1, 120, 12)
    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "E-commerce Shopping EMI",
            "Education EMI",
            "Home Appliances EMI",
            "Personal Loan EMI",
            "Vehicle EMI",
        ],
    )

# --- 2️⃣ Manual Encoding (FINAL = 27 categorical + 17 numeric = 44) ---

# Gender (2)
gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

# Marital Status (2)
marital_status_Married = 1 if marital_status == "Married" else 0
marital_status_Single = 1 if marital_status == "Single" else 0

# Education (5)
education_Graduate = 1 if education == "Graduate" else 0
education_High_School = 1 if education == "High School" else 0
education_Others = 1 if education == "Others" else 0
education_Post_Graduate = 1 if education == "Post Graduate" else 0
education_Professional = 1 if education == "Professional" else 0

# Employment Type (4)
employment_type_Government = 1 if employment_type == "Government" else 0
employment_type_Private = 1 if employment_type == "Private" else 0
employment_type_SelfEmployed = 1 if employment_type == "Self-employed" else 0
employment_type_Others = 1 if employment_type == "Others" else 0

# Company Type (4)
company_type_Small = 1 if company_type == "Small" else 0
company_type_Medium = 1 if company_type == "Medium" else 0
company_type_Large = 1 if company_type == "Large" else 0
company_type_MNC = 1 if company_type == "Multinational" else 0

# House Type (3)
house_type_Family = 1 if house_type == "Family" else 0
house_type_Own = 1 if house_type == "Own" else 0
house_type_Rented = 1 if house_type == "Rented" else 0

# Existing Loans (2)
existing_loans_No = 1 if existing_loans == "No" else 0
existing_loans_Yes = 1 if existing_loans == "Yes" else 0

# EMI Scenario (5)
emi_scenario_Ecommerce = 1 if emi_scenario == "E-commerce Shopping EMI" else 0
emi_scenario_Education = 1 if emi_scenario == "Education EMI" else 0
emi_scenario_Home = 1 if emi_scenario == "Home Appliances EMI" else 0
emi_scenario_Personal = 1 if emi_scenario == "Personal Loan EMI" else 0
emi_scenario_Vehicle = 1 if emi_scenario == "Vehicle EMI" else 0

# --- 3️⃣ Combine all features (exactly 44 total) ---
input_data = np.array([[
    # 17 numeric
    age, monthly_salary, years_of_employment, monthly_rent, family_size, dependents,
    school_fees, college_fees, travel_expenses, groceries_utilities, other_monthly_expenses,
    current_emi_amount, credit_score, bank_balance, emergency_fund, requested_amount, requested_tenure,

    # 27 categorical
    gender_female, gender_male,
    marital_status_Married, marital_status_Single,
    education_Graduate, education_High_School, education_Others, education_Post_Graduate, education_Professional,
    employment_type_Government, employment_type_Private, employment_type_SelfEmployed, employment_type_Others,
    company_type_Small, company_type_Medium, company_type_Large, company_type_MNC,
    house_type_Family, house_type_Own, house_type_Rented,
    existing_loans_No, existing_loans_Yes,
    emi_scenario_Ecommerce, emi_scenario_Education, emi_scenario_Home, emi_scenario_Personal, emi_scenario_Vehicle
]])

# --- 4️⃣ Predictions ---
colA, colB = st.columns(2)

with colA:
    if st.button("🔍 Predict EMI Eligibility"):
        pred = class_model.predict(input_data)[0]
        if pred == 0:
            st.warning("🚫 Not Eligible for EMI")
        elif pred == 1:
            st.info("⚠️ High Risk: EMI may be approved with conditions")
        else:
            st.success("✅ Eligible for EMI")

with colB:
    if st.button("💸 Predict Maximum EMI Amount"):
        max_emi = reg_model.predict(input_data)[0]
        st.metric("Estimated Maximum EMI (₹)", f"{max_emi:,.2f}")

st.markdown("---")
st.caption("Developed by EMIPredict AI • XGBoost | Streamlit | FinTech ML Platform")
