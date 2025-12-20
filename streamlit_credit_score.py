import streamlit as st
import pandas as pd
import cv2
import numpy as np
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Credit Score Classification", layout="wide")
# --- Streamlit App UI ---
st.title("💳 Credit Score Predictor")
col1, col2 = st.columns(2)
with col1:
    st.header("Estimate Your Credit Score")
    st.write("""
    Your credit score is one of the most important numbers in your financial life. 
    It can determine whether you can get financing and can mean the difference between 
    a preferential rate that saves you thousands of dollars or a more expensive loan.
    """)

with col2:
    st.image("https://ecm.capitalone.com/WCM/creditwise/cw-simulator-banner-b4.d-desktop.png")
# Form inputs
st.header("Enter Your Information")
st.write("Estimate your credit score in about 30 seconds. Just answer a few simple questions about your past credit usage:")

with st.form("credit_score_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100)
        annual_income = st.number_input("Annual Income", min_value=0)
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0)
        num_credit_cards = st.number_input("Number of Credit Cards", min_value=0)
        interest_rate = st.number_input("Interest Rate", min_value=0)
        num_loans = st.number_input("Number of Loans", min_value=0)
        delay_days = st.number_input("Delay from due date", min_value=0)
        num_delayed_payments = st.number_input("Number of Delayed Payments", min_value=0)
        
    with col2:
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0)
        credit_utilization = st.number_input("Credit Utilization Ratio", min_value=0)
        credit_history_age = st.number_input("Credit History Age", min_value=0)
        monthly_emi = st.number_input("Total EMI per month", min_value=0)
        monthly_investment = st.number_input("Amount invested monthly", min_value=0)
        payment_behavior = st.number_input("Payment Behaviour", min_value=0)
        monthly_balance = st.number_input("Monthly Balance", min_value=0)
        payment_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])

    submitted = st.form_submit_button("Calculate Credit Score")

if submitted:
    # Process credit mix
    credit_mix_map = {"Bad": 1, "Standard": 2, "Good": 3}
    credit_mix_value = credit_mix_map[credit_mix]
    
    # Process payment of min amount
    pma_nm = 1 if payment_min_amount == "NM" else 0
    pma_no = 1 if payment_min_amount == "No" else 0 
    pma_yes = 1 if payment_min_amount == "Yes" else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Annual_Income': [annual_income],
        'Num_Bank_Accounts': [num_bank_accounts],
        'Num_Credit_Card': [num_credit_cards],
        'Interest_Rate': [interest_rate],
        'Num_of_Loan': [num_loans],
        'Delay_from_due_date': [delay_days],
        'Num_of_Delayed_Payment': [num_delayed_payments],
        'Credit_Mix': [credit_mix_value],
        'Outstanding_Debt': [outstanding_debt],
        'Credit_Utilization_Ratio': [credit_utilization],
        'Credit_History_Age': [credit_history_age],
        'Total_EMI_per_month': [monthly_emi],
        'Amount_invested_monthly': [monthly_investment],
        'Payment_Behaviour': [payment_behavior],
        'Monthly_Balance': [monthly_balance],
        'PMA_NM': [pma_nm],
        'PMA_No': [pma_no],
        'PMA_Yes': [pma_yes]
    })

    # Scale the features
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(input_data), columns=input_data.columns)
    
    # Load model and make prediction
    model = xgb.Booster()
    model.load_model('model.h5')
    dmatrix = xgb.DMatrix(scaled_data)
    prediction = model.predict(dmatrix)

    # Show results
    st.header("Results")
    if prediction == 0:
        st.error("Credit Score: Poor")
        st.write("A poor credit score means that you are not eligible for application of loans")
    elif prediction == 1:
        st.warning("Credit Score: Standard")
        st.write("A standard credit score means that you will likely be eligible for application of small amount for loan")
    else:
        st.success("Credit Score: Good")
        st.write("A good credit score means that you will likely be eligible for application of large sum for loan")

# Additional information section
st.markdown("""
## What Is a Credit Score?

A credit score is tabulated by credit bureaus, who get information from the banks and companies you do business with about your financial payments. Your score is based on five basic things:

* How often you pay your bills on time and how often you're late (35% of score)
* How much you owe (10% of score)
* How many types of debts and credit lines you have (10% of score)
* Credit history length (15% of score)
* Number of recent credit inquiries (10% of score)

**Note:** Your income is not directly factored into your credit score.
""")
