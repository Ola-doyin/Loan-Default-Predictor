import streamlit as st
from datetime import date, datetime
import pandas as pd
import numpy as np
import random
import string
from customer_backend import predict_returning_customer
from models.custom_transformers import RiskClusterTransformer
import os


BASE_DIR = os.path.dirname(__file__)  # path of the current script
customer_data = pd.read_csv(os.path.join(BASE_DIR, "customer_data.csv"), index_col="customerid")

st.title("Loan Application Portal")

# Choose customer type
customer_type = st.segmented_control(
    "Are you a:",
    ["New Customer", "Old Customer"]
)




#### New Customer Section ######

if customer_type == "New Customer":
    st.subheader("New Customer Loan Application Form")

    username = st.text_input("Choose a Username (will be your Customer ID)")
    if username:
        customerid = st.text_input("customerid", value='new' + username + ''.join(random.choices(string.digits, k=30))[:32], disabled=True)
    
    loan_amount = st.number_input("Loan Amount (₦)", min_value=0, key="loan_amount_new", step=100)
    termdays = st.selectbox("Select Term (Days)", [15, 30], key="termdays_new")

    # Interest rate calculation
    if termdays == 15:
        interest_rate = 15.0 if loan_amount <= 20000 else 12.0
    else:  # 30 days
        if loan_amount <= 10000:
            interest_rate = 30.0
        elif loan_amount <= 20000:
            interest_rate = 22.5
        elif loan_amount <= 30000:
            interest_rate = 15.0
        elif loan_amount <= 40000:
            interest_rate = 10.0
        else:
            interest_rate = 5.0

    total_due = loan_amount * (1 + interest_rate / 100)
    st.write(f"**Interest Rate:** {interest_rate}%")
    st.write(f"**Total Due:** ₦{total_due:,.2f}")

    # Step 2: Form for other details
    with st.form("loan_application_form"):

        # Date of Birth & Age
        dob = st.date_input("Date of Birth", min_value=date(1900, 1, 1), max_value=date.today())
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        bank_account = st.selectbox("Bank Account Type", [" ", "Savings", "Current", "Fixed Deposit", "Other"])
        bank_account_type = ("Savings" if bank_account == "Savings" else "Other" if bank_account else "Unknown")

        bank_name_clients = st.selectbox("Bank Name", [" ", "Access Bank", "Diamond Bank", "EcoBank", "FCMB", "Fidelity Bank", "First Bank", "GT Bank", 
                                                       "Heritage Bank", "Keystone Bank", "Skye Bank", "Stanbic IBTC", "Standard Chartered", "Sterling Bank", 
                                                       "UBA", "Union Bank", "Unity Bank", "Wema Bank", "Zenith Bank", "Other"])
        
        employment_status_clients = st.selectbox("Employment Status", [" ", "Permanent", "Self-Employed", "Contract", "Student", "Retired", "Unemployed"])

        state = st.selectbox("State of Residence", [" ", "Abroad", "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bayelsa", "Benue", "Bauchi", "Borno", 
                                                    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Federal Capital Territory", "Gombe", "Imo", 
                                                    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", 
                                                    "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Taraba", "Zamfara"])
        
        referred = st.text_input("Referred By: (optional)")

        submitted = st.form_submit_button("Submit Application")

    if submitted:
        if (not username.strip() or not bank_account.strip() or not bank_name_clients.strip() 
        or not employment_status_clients.strip() or not state.strip()):
            st.error("Please complete biodata form.")
        
        else:
            st.success("Your loan application was submitted successfully!")

            new_customer_input = {
                "customerid": customerid,
                "loannumber": 1,
                "loanamount": loan_amount,
                "totaldue": total_due,
                "termdays": termdays,
                "bank_account_type": bank_account_type or "Unknown",
                "bank_name_clients": bank_name_clients or "Unknown",
                "employment_status_clients": employment_status_clients or "Unknown",
                "credit_score": 2,
                "debt_to_loan_ratio": float(total_due / loan_amount if loan_amount > 0 else 0),
                "unusual_large": 1 if loan_amount > 0.5*(customer_data.loanamount.median()) else 0,
                "state": state or "Unknown",
                "age": age,
                "referred": 1 if referred.strip() else 0,
                "last_loan_amount": 0,
                "loan_growth_trend": 0,

                # "avg_amount": 0,
                # "std_amount": 0,
                # "avg_total_due": 0,
                # "avg_term_days": 0,
                # "std_term_days": 0,
                # "avg_approval_time": 0,
                # "avg_payout_time": 0,
                # "prct_duedate_default": 0,
                # "max_repay_delay": 0,
                # "avg_loan_gap": 0,
                # "acct_age_years": 0,
                # "time_btwn_last_loan": 0,
                # "const_default": 1
                
            }

            # Fill missing columns with medians
            for col in customer_data.columns:
                if col not in new_customer_input:
                    new_customer_input[col] = customer_data[col].median() if np.issubdtype(customer_data[col].dtype, np.number) else "Unknown"

            # Predict
            pred, conf = predict_returning_customer(new_customer_input)
                        
            # Map output to readable label
            st.subheader("Admin End Prediction")
            # st.text_input("Customer ID", value=cust_id, disabled=True)
            
            # Create two columns for amount and term
            col1, col2 = st.columns(2)

            with col1:
                st.text_input("Loan Amount (₦)", value=f"₦{loan_amount:,.2f}", disabled=True)

            with col2:
                st.text_input("Term Days", value=termdays, disabled=True)

            label_map = {0: "Eligible", 1: "Needs Review"}
            result_label = label_map.get(pred, "Unknown")
            st.metric("Prediction", result_label)
            st.metric("Confidence", f"{conf}%")
        





###### Returning Customer Section ######
elif customer_type == "Old Customer":
    st.subheader("Returning Customer")

    # Step 1: Enter Customer ID
    cust_id = st.text_input("Enter your Customer ID")

    if cust_id:
        if cust_id in customer_data.index:
            cust_row = customer_data.loc[cust_id]

            # Loan details (editable) OUTSIDE the form
            loan_amount = st.number_input("Loan Amount (₦)", min_value=0, value=0, key="loan_amount_old", step=100)
            termdays = st.selectbox("Select Term (Days)", [15, 30, 60, 90], index=0, key="termdays_old")

            # Auto recalculation updates live
            interest_rate = 0.15
            total_due = loan_amount * (1 + interest_rate)
            st.write(f"Interest Rate: {interest_rate*100:.1f}%")
            st.write(f"Total Due: ₦{total_due:,.2f}")

            # Step 2: Form for other details + final submit
            with st.form("old_customer_form"):
                # Auto-populated fields
                loan_num = st.text_input("Loan Number", value=(cust_row["loannumber"] + 1), disabled=True)
                st.text_input("Age", value=(cust_row["age"]), disabled=True)
                bank_account_type = st.text_input("Bank Account Type", value=cust_row["bank_account_type"], disabled=True)
                bank_name_clients = st.text_input("Bank Name", value=cust_row["bank_name_clients"], disabled=True)
                employment_status_clients = st.text_input("Employment Status", value=cust_row["employment_status_clients"], disabled=True)
                state = st.text_input("State of Residence", value=cust_row["state"], disabled=True)
                age = cust_row["age"]

                # Editable referral
                referred = st.text_input("Referred By (optional)")

                # ✅ The only real submit
                submitted = st.form_submit_button("Submit Application")

                if submitted:
                    new_customer_input = {
                        "customerid": str(cust_id),
                        "loannumber": int(loan_num),
                        "loanamount": float(loan_amount),
                        "totaldue": float(total_due),
                        "termdays": int(termdays),
                        "bank_account_type": str(bank_account_type or "Unknown"),
                        "bank_name_clients": str(bank_name_clients or "Unknown"),
                        "employment_status_clients": str(employment_status_clients or "Unknown"),
                        "state": str(state or "Unknown"),
                        "age": int(age),   
                        "debt_to_loan_ratio": float(total_due / loan_amount if loan_amount > 0 else 0),
                        "referred": 1 if referred.strip() else 0
                    }

                    # Pull the rest of the fields from dataset
                    cust_row_dict = customer_data.loc[cust_id].to_dict()

                    # Merge inputs into the full row (overwriting if duplicate keys exist)
                    full_customer_data = {**cust_row_dict, **new_customer_input}

                    # Send to Returning_customer.py
                    # processed_features = Returning_customer.prepare_returning_customer_features(full_customer_data)

                    st.success("Your loan application was submitted successfully!")
                    # st.write("Processed returning customer features:", processed_features)

                    # later -> send to model
                    pred, conf = predict_returning_customer(full_customer_data)
                    
                    # Map output to readable label
                    st.subheader("Admin End Prediction")
                    st.text_input("Customer ID", value=cust_id, disabled=True)
                    
                    # Create two columns for amount and term
                    col1, col2 = st.columns(2)

                    with col1:
                        st.text_input("Loan Amount (₦)", value=f"₦{loan_amount:,.2f}", disabled=True)

                    with col2:
                        st.text_input("Term Days", value=termdays, disabled=True)

                    label_map = {0: "Auto-approve", 1: "Needs Review"}
                    result_label = label_map.get(pred, "Unknown")
                    st.metric("Prediction", result_label)
                    st.metric("Confidence", f"{conf}%")

        else:
            st.error("Customer ID not found. Please try again.")
