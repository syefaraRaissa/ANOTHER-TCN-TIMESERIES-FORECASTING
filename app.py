import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from keras.models import load_model
from tcn import TCN
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prediksi Tag Value", layout="centered")
st.title("🔮 Prediksi Tag Value 10 Menit ke Depan (Setiap 10 Detik)")

# Constants
WINDOW_SIZE = 60      # Sesuai model: 60 input (10 detik * 60 = 10 menit sebelumnya)
FUTURE_STEPS = 60     # Output: 60 langkah ke depan (10 menit)

# Fungsi cache load model dan scaler
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("my_model.h5", custom_objects={"TCN": TCN})
        scaler = joblib.load("scalercp.joblib")
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau scaler: {e}")
        st.stop()

# Load model dan scaler
model, scaler = load_artifacts()

# Upload data CSV
uploaded_file = st.file_uploader("📂 Upload file CSV dengan kolom 'ddate' dan 'tag_value'", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate').reset_index(drop=True)

        st.subheader("📊 Data Terakhir")
        st.dataframe(df.tail(5))

        if len(df) < WINDOW_SIZE:
            st.error(f"❌ Data kurang dari {WINDOW_SIZE} baris. Tidak bisa memprediksi.")
        else:
            # Ambil WINDOW_SIZE nilai terakhir
            last_values = df['tag_value'].values[-WINDOW_SIZE:]
            scaled_input = scaler.transform(last_values.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)

            forecast = []
            current_input = scaled_input.copy()

            for _ in range(FUTURE_STEPS):
                pred = model.predict(current_input, verbose=0)[0, 0]
                forecast.append(pred)
                # Update input window
                current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

            # Kembalikan ke skala asli
            forecast_inv = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            last_timestamp = df['ddate'].iloc[-1]
            forecast_timestamps = [last_timestamp + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

            result_df = pd.DataFrame({
                'Datetime': forecast_timestamps,
                'Prediksi Tag Value': forecast_inv.flatten()
            })

            st.subheader("📈 Grafik Prediksi")
            st.line_chart(result_df.set_index("Datetime"))

            st.subheader("📋 Tabel Prediksi")
            st.dataframe(result_df)

            # Evaluasi jika tersedia data aktual (60 data setelah input)
            if len(df) >= WINDOW_SIZE + FUTURE_STEPS:
                actual_future = df['tag_value'].values[-FUTURE_STEPS:]
                mae = mean_absolute_error(actual_future, forecast_inv)
                rmse = np.sqrt(mean_squared_error(actual_future, forecast_inv))

                st.subheader("📉 Evaluasi Model")
                st.markdown(f"""
                - **MAE** (Mean Absolute Error): `{mae:.4f}`
                - **RMSE** (Root Mean Squared Error): `{rmse:.4f}`
                """)
            else:
                st.info("📎 Tidak cukup data aktual untuk evaluasi (butuh 60 data setelah input).")

    except Exception as e:
        st.error(f"❌ Terjadi error saat memproses file: {e}")
