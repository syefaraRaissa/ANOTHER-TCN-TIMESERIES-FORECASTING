import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tcn import TCN
import matplotlib.pyplot as plt
from datetime import timedelta

# Load model dan scaler
model = load_model("my_model.h5", custom_objects={'TCN': TCN})
scaler = joblib.load("scalercp.joblib")

# Fungsi untuk buat window input
def create_window(data, window_size=60):
    scaled = scaler.transform(data[['tag_value']])
    return scaled[-window_size:].reshape(1, window_size, 1)

# Judul aplikasi
st.title("🔮 Prediksi Tag Value 10 Menit ke Depan (TCN Model)")

# Upload file CSV baru
uploaded_file = st.file_uploader("Unggah file CSV (dengan kolom 'ddate' dan 'tag_value')", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['ddate'] = pd.to_datetime(df['ddate'])

    st.subheader("Data Terakhir")
    st.write(df.tail())

    # Prediksi 10 menit ke depan (60 titik data per 10 detik)
    window_size = 60
    if len(df) >= window_size:
        last_window = create_window(df, window_size=window_size)
        forecast = []

        for _ in range(60):  # 10 menit = 60 langkah (10 detik sekali)
            pred_scaled = model.predict(last_window, verbose=0)
            forecast.append(pred_scaled[0, 0])
            last_window = np.append(last_window[:, 1:, :], [[pred_scaled]], axis=1)

        # Balikkan ke skala asli
        forecast_inv = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

        # Buat tanggal untuk prediksi
        last_date = df['ddate'].iloc[-1]
        forecast_dates = [last_date + timedelta(seconds=10 * (i+1)) for i in range(60)]

        # Tampilkan hasil
        result_df = pd.DataFrame({
            "Waktu": forecast_dates,
            "Prediksi Tag Value": forecast_inv.flatten()
        })

        st.subheader("📈 Hasil Prediksi 10 Menit ke Depan")
        st.line_chart(result_df.set_index("Waktu"))

        with st.expander("Lihat Data Prediksi"):
            st.write(result_df)
    else:
        st.warning("Data tidak cukup untuk prediksi. Minimal 60 baris.")
