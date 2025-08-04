import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from tcn import TCN

st.set_page_config(page_title="Prediksi Tag Value", layout="centered")
st.title("🔮 Prediksi Tag Value 10 Menit Ke Depan (Setiap 10 Detik)")

# Load model dan scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model('my_model.h5', custom_objects={'TCN': TCN})  # atau ganti dengan 'my_model.keras'
    scaler = joblib.load('scaler.joblib')  # atau ganti dengan 'scaler (2).pkl'
    return model, scaler

model, scaler = load_model_and_scaler()

# Fungsi prediksi 60 langkah ke depan
def predict_next_60_steps(input_sequence, model, scaler):
    scaled_input = scaler.transform(input_sequence)
    scaled_input = scaled_input.reshape(1, scaled_input.shape[0], scaled_input.shape[1])
    predicted_scaled = model.predict(scaled_input)[0]
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
    return predicted.flatten()

# Upload data
uploaded_file = st.file_uploader("📤 Upload data sensor (CSV, minimal 60 baris)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if 'tag_value' not in df.columns:
        st.error("❌ Kolom 'tag_value' tidak ditemukan.")
    elif len(df) < 60:
        st.error("❌ Data harus memiliki minimal 60 baris.")
    else:
        df = df.tail(60)
        input_sequence = df[['tag_value']].values
        pred = predict_next_60_steps(input_sequence, model, scaler)

        future_times = [pd.Timestamp.now() + timedelta(seconds=10 * (i + 1)) for i in range(60)]
        result_df = pd.DataFrame({'timestamp': future_times, 'predicted_tag_value': pred})
        
        st.success("✅ Prediksi berhasil!")
        st.line_chart(result_df.set_index('timestamp'))
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download hasil prediksi", csv, "prediksi_tag_value.csv", "text/csv")

