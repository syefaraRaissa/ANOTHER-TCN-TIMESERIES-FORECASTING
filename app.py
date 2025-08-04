import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from tcn import TCN

# Judul
st.title("🔮 Prediksi Tag Value 10 Menit Ke Depan (per 10 Detik)")

# Parameter model
WINDOW_SIZE = 60     # Jumlah titik data sebelumnya (60 x 10 detik = 10 menit)
FUTURE_STEPS = 60    # Jumlah langkah prediksi (10 detik x 60 = 10 menit)

# Cache model dan scaler
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("my_model.keras", compile=False, custom_objects={"TCN": TCN})
        scaler = joblib.load("scalercp.joblib")  # ← pastikan ini sama seperti di Colab
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
            # Ambil WINDOW_SIZE terakhir
            last_values = df['tag_value'].values[-WINDOW_SIZE:]

            # Normalisasi
            scaled_input = scaler.transform(last_values.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)

            forecast_scaled = []
            current_input = scaled_input.copy()

            noise_std = 0.002  # Sesuai Colab
            for _ in range(FUTURE_STEPS):
                next_pred = model.predict(current_input, verbose=0)
                noisy_pred = next_pred + np.random.normal(0, noise_std, size=next_pred.shape)
                forecast_scaled.append(noisy_pred[0, 0])
                current_input = np.append(current_input[:, 1:, :], [[[noisy_pred[0, 0]]]], axis=1)


            # Inverse transform
            forecast_actual = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

            # Buat waktu prediksi
            last_time = df['ddate'].iloc[-1]
            future_times = [last_time + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

            forecast_df = pd.DataFrame({
                "Datetime": future_times,
                "Prediksi Tag Value": forecast_actual
            })

            st.subheader("📈 Grafik Prediksi")
            st.line_chart(forecast_df.set_index("Datetime"))

            st.subheader("📋 Tabel Prediksi")
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"❌ Error saat memproses data: {e}")
