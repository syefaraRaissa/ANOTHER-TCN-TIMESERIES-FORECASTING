import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from tcn import TCN

# Judul
st.title("🔮 Prediksi Tag Value 10 Menit Ke Depan (Tanpa Noise)")

# Parameter model
WINDOW_SIZE = 60     # Jumlah titik data sebelumnya (60 x 10 detik = 10 menit)
FUTURE_STEPS = 60    # Jumlah langkah prediksi (10 detik x 60 = 10 menit)

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
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate').reset_index(drop=True)

        st.subheader("📊 Data Terakhir:")
        st.dataframe(df.tail(5))

        if len(df) < WINDOW_SIZE:
            st.error(f"❌ Data kurang. Minimal {WINDOW_SIZE} baris diperlukan.")
        else:
            # Ambil WINDOW_SIZE terakhir dari data
            last_values = df['tag_value'].values[-WINDOW_SIZE:]
            scaled_input = scaler.transform(last_values.reshape(-1, 1))

            forecast_scaled = []
            last_window = scaled_input.copy()

            for _ in range(FUTURE_STEPS):
                input_data = last_window.reshape((1, WINDOW_SIZE, 1))
                next_pred = model.predict(input_data, verbose=0)[0, 0]
                forecast_scaled.append(next_pred)

                # Update window dengan prediksi (tanpa noise)
                last_window = np.append(last_window[1:], [[next_pred]], axis=0)

            # Inverse transform hasil prediksi
            forecast_actual = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

            # Buat waktu prediksi
            last_time = df['ddate'].iloc[-1]
            future_times = [last_time + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

            forecast_df = pd.DataFrame({
                "Tanggal": future_times,
                "Prediksi Tag Value (Tanpa Noise)": forecast_actual
            })

            st.subheader("📈 Grafik Prediksi (Tanpa Noise)")
            st.line_chart(forecast_df.set_index("Tanggal"))

            st.subheader("📋 Tabel Prediksi (Tanpa Noise)")
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"❌ Error saat memproses data: {e}")
