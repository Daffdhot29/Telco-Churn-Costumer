import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Rekomendasi Target Offer Telco", layout="centered")
st.title("Model Trial - Rekomendasi Target Offer Telco")

MODEL_PATH = "model_rf.joblib"


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model '{path}' tidak ditemukan.")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

model = load_model(MODEL_PATH)
if model is None:
    st.stop()

st.subheader("Masukkan Data Pengguna")

plan_type = st.selectbox("Plan Type", ["prepaid", "postpaid"])
device_brand = st.selectbox("Device Brand", 
    ["realme", "vivo", "xiaomi", "apple", "huawei", "oppo", "samsung"]
)

avg_data_usage_gb = st.number_input("Average Data Usage (GB)", 0.0, 10000.0, 5.0)
pct_video_usage = st.number_input("Video Usage (%)", 0.0, 100.0, 30.0)
avg_call_duration = st.number_input("Average Call Duration", 0.0, 10000.0, 15.0)
sms_freq = st.number_input("SMS Frequency", 0, 50000, 10)
monthly_spend = st.number_input("Monthly Spend", 0.0, 1000000.0, 50.0)
topup_freq = st.number_input("Topup Frequency", 0, 500, 2)
travel_score = st.number_input("Travel Score", 0.0, 100.0, 10.0)
complaint_count = st.number_input("Complaint Count", 0, 50, 0)


X_input = pd.DataFrame([{
    "plan_type": plan_type.lower(),
    "device_brand": device_brand.lower(),
    "avg_data_usage_gb": avg_data_usage_gb,
    "pct_video_usage": pct_video_usage,
    "avg_call_duration": avg_call_duration,
    "sms_freq": sms_freq,
    "monthly_spend": monthly_spend,
    "topup_freq": topup_freq,
    "travel_score": travel_score,
    "complaint_count": complaint_count
}])


if st.button("🚀 Prediksi"):
    try:
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        st.subheader("🎯 Hasil Prediksi Offer")
        st.success(str(pred))

        st.subheader("📊 Probabilitas Offer")
        st.write(pd.DataFrame({
            "Offer": model.classes_,
            "Prob (%)": (proba * 100).round(2)
        }))
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
