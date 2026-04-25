import streamlit as st
import requests
import time

st.title("📊 Churn Risk Dashboard (Async)")

API_BASE = "http://api:8000"
PREDICT_ASYNC = f"{API_BASE}/predict_async"
RESULT_ENDPOINT = f"{API_BASE}/result"

st.write("Enter a minimal subset of features (must match training schema).")
# --- Full input form ---
with st.expander("Input features (expand to enter full record)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("gender", ["Female", "Male"], index=0)
        SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1], index=0)
        Dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
        tenure = st.slider("tenure", 0, 72, 12)
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
        PaperlessBilling = st.selectbox("PaperlessBilling", ["No", "Yes"], index=1)
        MonthlyCharges = st.number_input("MonthlyCharges", value=70.0)
        TotalRevenue = st.number_input("TotalRevenue", value=1000.0)

    with col2:
        InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], index=0)
        OnlineSecurity = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"], index=0)
        OnlineBackup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"], index=0)
        DeviceProtection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"], index=0)
        TechSupport = st.selectbox("TechSupport", ["No", "Yes", "No internet service"], index=0)
        StreamingTV = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"], index=0)
        StreamingMovies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"], index=0)

    # Additional numeric fields
    col3, col4 = st.columns(2)
    with col3:
        TotalDayMinutes = st.number_input("TotalDayMinutes", value=0.0)
        TotalEveMinutes = st.number_input("TotalEveMinutes", value=0.0)
        TotalNightMinutes = st.number_input("TotalNightMinutes", value=0.0)
        TotalIntlMinutes = st.number_input("TotalIntlMinutes", value=0.0)
        TotalDayCalls = st.number_input("TotalDayCalls", value=0)
        TotalEveCalls = st.number_input("TotalEveCalls", value=0)

    with col4:
        TotalNightCalls = st.number_input("TotalNightCalls", value=0)
        TotalIntlCalls = st.number_input("TotalIntlCalls", value=0)
        TotalCall = st.number_input("TotalCall", value=0)
        NumbervMailMessages = st.number_input("NumbervMailMessages", value=0)
        CustomerServiceCalls = st.number_input("CustomerServiceCalls", value=0)

    # Phone / other
    PhoneService = st.selectbox("PhoneService", ["No", "Yes"], index=1)
    MultipleLines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"], index=0)
    InternationalPlan = st.selectbox("InternationalPlan", ["No", "Yes"], index=0)
    VoiceMailPlan = st.selectbox("VoiceMailPlan", ["No", "Yes"], index=0)
    PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
    MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unknown"], index=0)

    payload = {
        "data": {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Dependents": Dependents,
            "tenure": int(tenure),
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "MonthlyCharges": float(MonthlyCharges),
            "TotalRevenue": float(TotalRevenue),
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "TotalDayMinutes": float(TotalDayMinutes),
            "TotalEveMinutes": float(TotalEveMinutes),
            "TotalNightMinutes": float(TotalNightMinutes),
            "TotalIntlMinutes": float(TotalIntlMinutes),
            "TotalDayCalls": int(TotalDayCalls),
            "TotalEveCalls": int(TotalEveCalls),
            "TotalNightCalls": int(TotalNightCalls),
            "TotalIntlCalls": int(TotalIntlCalls),
            "TotalCall": int(TotalCall),
            "NumbervMailMessages": int(NumbervMailMessages),
            "CustomerServiceCalls": int(CustomerServiceCalls),
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternationalPlan": InternationalPlan,
            "VoiceMailPlan": VoiceMailPlan,
            "PaymentMethod": PaymentMethod,
            "MaritalStatus": MaritalStatus,
        }
    }

if st.button("Submit"):
    try:
        resp = requests.post(PREDICT_ASYNC, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        st.error(f"Failed to submit task: {exc}")
    else:
        data = resp.json()
        task_id = data.get("task_id")
        if not task_id:
            st.error(f"Unexpected response from API: {data}")
        else:
            st.info(f"Task submitted: {task_id}. Polling for result...")
            placeholder = st.empty()
            progress = st.progress(0)
            timeout_seconds = 30
            interval = 1
            start = time.time()
            step = 0
            while time.time() - start < timeout_seconds:
                try:
                    r = requests.get(f"{RESULT_ENDPOINT}/{task_id}", timeout=5)
                    if r.ok:
                        j = r.json()
                        status = j.get("status")
                        if status == "done":
                            placeholder.success("Result ready")
                            # show result object
                            result = j.get("result")
                            if isinstance(result, dict):
                                st.json(result)
                            else:
                                st.write(result)
                            break
                        else:
                            placeholder.info("Processing...")
                    else:
                        placeholder.warning(f"Status check failed: {r.status_code}")
                except Exception:
                    placeholder.warning("Error checking status; retrying...")

                step += 1
                progress.progress(int(min(100, (step * interval) / timeout_seconds * 100)))
                time.sleep(interval)

            else:
                st.warning("Timed out waiting for result. Use the Task ID to query later.")

# --- Data exploration / charts ---
st.header("Data Explorer")
import os
import pandas as _pd
# Resolve data path relative to this script's location so it works inside Docker.
base_dir = os.path.dirname(__file__)  # app/ directory inside container
data_path = os.path.abspath(os.path.join(base_dir, "..", "data", "data.csv"))
if os.path.exists(data_path):
    try:
        df = _pd.read_csv(data_path)
        st.subheader("Preview")
        st.dataframe(df.head())
        st.subheader("Summary statistics")
        st.write(df.describe(include='all'))

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        st.subheader("Numeric feature distributions")
        num_choice = st.selectbox("Choose numeric feature", [None] + numeric_cols)
        if num_choice:
            series = df[num_choice].dropna()
            bins = 20
            counts = series.groupby(_pd.cut(series, bins=bins)).size()
            st.bar_chart(counts)

        st.subheader("Categorical feature counts")
        cat_choice = st.selectbox("Choose categorical feature", [None] + cat_cols)
        if cat_choice:
            vc = df[cat_choice].fillna("(missing)").value_counts().nlargest(50)
            st.bar_chart(vc)

        st.subheader("Correlation (numeric)")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            st.dataframe(corr)
    except Exception as e:
        st.warning(f"Failed to load data for explorer: {e}")
else:
    st.info("No data file found at ../data/data.csv — add your dataset to enable charts.")
