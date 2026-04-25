import streamlit as st
import requests
import json

st.title("📊 Churn Risk Dashboard (Async)")

api_url = "http://api:8000/predict_async"

st.write("Enter a minimal subset of features (must match training schema).")
tenure = st.slider("tenure", 0, 72, 12)
monthly = st.number_input("MonthlyCharges", value=70.0)

payload = {
    "data": {
        "tenure": tenure,
        "MonthlyCharges": monthly
    }
}

if st.button("Submit"):
    r = requests.post(api_url, json=payload)
    st.write(r.json())
