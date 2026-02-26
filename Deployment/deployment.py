import os

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn 

import joblib
import streamlit as st

# Load the model 
model=joblib.load('/workspaces/Customer-Churn-Prediction/Models/customer_churn_model.pkl')


# Load feature names 
columns=joblib.load('/workspaces/Customer-Churn-Prediction/Models/feature_columns.pkl')


# Outline 
st.set_page_config(page_title='Customer Churn Prediction',layout='wide')

st.title('Customer Churn Prediction')
st.write('Enter customer details to predict if they are likely to churn or not')

# User inputs
gender=st.selectbox("Gender", ["Male","Female"])
senior=st.selectbox("Senior Citizen", [0,1])
partner=st.selectbox("Partner", ["Yes","No"])
dependents=st.selectbox("Dependents", ["Yes","No"])

tenure=st.number_input("Tenure (months)", 0, 72)

phone=st.selectbox("Phone Service", ["Yes","No"])
multiple=st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

internet=st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

online_security=st.selectbox("Online Security", ["Yes","No","No internet service"])
online_backup=st.selectbox("Online Backup", ["Yes","No","No internet service"])
device_protection=st.selectbox("Device Protection", ["Yes","No","No internet service"])
tech_support=st.selectbox("Tech Support", ["Yes","No","No internet service"])
streaming_tv=st.selectbox("Streaming TV", ["Yes","No","No internet service"])
streaming_movies=st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

contract=st.selectbox("Contract", ["Month-to-month","One year","Two year"])
paperless=st.selectbox("Paperless Billing", ["Yes","No"])
payment=st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly=st.number_input("Monthly Charges")
total=st.number_input("Total Charges")

# Prepare input data for prediction
input_df=pd.DataFrame([{
    'gender':gender,
    'SeniorCitizen':senior,
    'Partner':partner,
    'Dependents':dependents,
    'tenure':tenure,
    'PhoneService':phone,
    'MultipleLines':multiple,
    'InternetService':internet,
    'OnlineSecurity':online_security,
    'OnlineBackup':online_backup,
    'DeviceProtection':device_protection,
    'TechSupport':tech_support,
    'StreamingTV':streaming_tv,
    'StreamingMovies':streaming_movies,
    'Contract':contract,
    'PaperlessBilling':paperless,
    'PaymentMethod':payment,
    'MonthlyCharges':monthly,
    'TotalCharges':total
}])

input_df=pd.get_dummies(input_df)
input_df=input_df.reindex(columns=columns,fill_value=0)

# Predictions
if st.button("Predict Churn"):

    probability=model.predict_proba(input_df)[0][1]
    threshold=0.3  
    prediction="Churn" if probability>threshold else "No Churn"

    st.subheader("Prediction Result")
    st.write("Churn Probability:",round(probability,3))
    st.write("Final Prediction:",prediction)

    # Risk Level
    if probability<0.3:
        st.success("Low Risk Customer")
    elif probability<0.6:
        st.warning("Medium Risk Customer")
    else:
        st.error("High Risk Customer")
