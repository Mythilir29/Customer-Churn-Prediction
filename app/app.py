import streamlit as st
import pandas as pd
import joblib
import os

# Load model
MODEL_PATH = os.path.join("models", "churn_pipeline.pkl")
model = joblib.load(MODEL_PATH)

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details to predict churn")

# Sidebar inputs
st.sidebar.header("Customer Input Features")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Senior_Citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

Tenure_Months = st.sidebar.slider("Tenure Months", 0, 72, 12)

Phone_Service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
Multiple_Lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

Internet_Service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

Online_Security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
Online_Backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
Device_Protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
Tech_Support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])

Streaming_TV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
Streaming_Movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
Paperless_Billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

Payment_Method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

Monthly_Charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
Total_Charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Prediction button
if st.button("🔍 Predict Churn"):

    # IMPORTANT: DataFrame (NOT numpy array)
    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Senior_Citizen": Senior_Citizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "Tenure_Months": Tenure_Months,
        "Phone_Service": Phone_Service,
        "Multiple_Lines": Multiple_Lines,
        "Internet_Service": Internet_Service,
        "Online_Security": Online_Security,
        "Online_Backup": Online_Backup,
        "Device_Protection": Device_Protection,
        "Tech_Support": Tech_Support,
        "Streaming_TV": Streaming_TV,
        "Streaming_Movies": Streaming_Movies,
        "Contract": Contract,
        "Paperless_Billing": Paperless_Billing,
        "Payment_Method": Payment_Method,
        "Monthly_Charges": Monthly_Charges,
        "Total_Charges": Total_Charges
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer WILL CHURN")
    else:
        st.success("✅ Customer will NOT churn")

    st.metric("Churn Probability", f"{probability:.2f}")