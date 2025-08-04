import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from tcn import TCN

st.set_page_config(page_title="Prediksi Tag Value", layout="wide")
st.title("🔮 Prediksi Tag Value 10 Menit Ke Depan (per 10 Detik)")

# Load model dan scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("my_model.keras", custom_objects={'TCN': TCN})
    scaler = joblib.load("scalercp.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Upload data CSV
uploaded_file = st.file_uploader("📥 Upload data CSV (minimal 60 baris)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'tag_value' not in df.columns:
        st.error("❌ Kolom 'tag_value' tidak ditemukan. Harap pastikan file memiliki kolom 'tag_value'.")
    else:
        st.success("✅ File berhasil dibaca!")
        st.write("Contoh data terakhir:")
        st.dataframe(df.tail())

        sequence_length = 60

        if len(df) < sequence_length:
            st.warning("⚠️ Data kurang dari 60 baris. Harap upload data minimal 10 menit (60 baris per 10 detik).")
        else:
            # Ambil 60 data terakhir
            last_sequence = df['tag_value'].values[-sequence_length:].reshape(-1, 1)

            # Normalisasi dan reshape
            scaled_sequence = scaler.transform(last_sequence)
            input_sequence = scaled_sequence.reshape(1, sequence_length, 1)

            # Prediksi
            prediction_scaled = model.predict(input_sequence)
            prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

            # Buat waktu prediksi
            if 'timestamp' in df.columns:
                last_time = pd.to_datetime(df['timestamp'].iloc[-1])
                time_range = pd.date_range(start=last_time + timedelta(seconds=10), periods=60, freq="10S")
            else:
                time_range = [f"Step {i+1}" for i in range(60)]

            # Tampilkan hasil
            pred_df = pd.DataFrame({
                "Waktu": time_range,
                "Prediksi Tag Value": prediction
            })

            st.subheader("📈 Hasil Prediksi (60 langkah ke depan)")
            st.line_chart(pred_df.set_index("Waktu"))

            with st.expander("📋 Tabel Prediksi"):
                st.dataframe(pred_df)
