import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from tcn import TCN
import matplotlib.pyplot as plt

# Judul
st.title("🔮 Prediksi Tag Value 10 Menit Ke Depan")

# Parameter model
WINDOW_SIZE = 60     # 10 menit terakhir (60 x 10 detik)
FUTURE_STEPS = 60    # Prediksi 10 menit ke depan

# Cache model dan scaler
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("my_model.keras", compile=False, custom_objects={"TCN": TCN})
        scaler = joblib.load("scalercp.joblib")
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_artifacts()

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        # Pastikan kolom waktu benar
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate').reset_index(drop=True)

        st.subheader("📊 Data Terakhir:")
        st.dataframe(df.tail(5))

        if len(df) < WINDOW_SIZE:
            st.error(f"❌ Data kurang. Minimal {WINDOW_SIZE} baris diperlukan.")
        else:
            # Ambil WINDOW_SIZE terakhir dan skala
            last_values = df['tag_value'].values[-WINDOW_SIZE:]
            scaled_input = scaler.transform(last_values.reshape(-1, 1))

            forecast_scaled = []
            last_window = scaled_input.copy()

            # Loop prediksi
            for _ in range(FUTURE_STEPS):
                input_data = last_window.reshape((1, WINDOW_SIZE, 1))
                next_pred = model.predict(input_data, verbose=0)[0, 0]
                forecast_scaled.append(next_pred)
                last_window = np.append(last_window[1:], [[next_pred]], axis=0)

            # Kembalikan ke nilai asli
            forecast_actual = scaler.inverse_transform(
                np.array(forecast_scaled).reshape(-1, 1)
            ).flatten()

            # Buat waktu prediksi
            last_time = df['ddate'].iloc[-1]
            future_times = [last_time + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

            forecast_df = pd.DataFrame({
                "Tanggal": future_times,
                "Prediksi Tag Value": forecast_actual
            })

            # ---- Plot dengan Matplotlib ----
            st.subheader("📈 Grafik Prediksi")
            plt.figure(figsize=(8, 5))

            # Data historis
            plt.plot(df['ddate'].iloc[-WINDOW_SIZE:], df['tag_value'].iloc[-WINDOW_SIZE:], 
                     label='Data Historis', color='blue')

            # Data prediksi
            plt.plot(forecast_df['Tanggal'], forecast_df['Prediksi Tag Value'], 
                     label='Prediksi', color='red')

            plt.legend()
            plt.xlabel("Waktu")
            plt.ylabel("Tag Value")
            plt.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # ---- Tabel prediksi ----
            st.subheader("📋 Tabel Prediksi")
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"❌ Error saat memproses data: {e}")
