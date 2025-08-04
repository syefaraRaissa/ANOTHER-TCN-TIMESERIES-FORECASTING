import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model
from tcn import TCN
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prediksi Tag Value", layout="centered")
st.title("🔮 Prediksi Tag Value 10 Menit Ke Depan (Setiap 10 Detik)")

# Fungsi caching untuk load model dan scaler
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("my_model.keras", compile=False, custom_objects={"TCN": TCN})
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_artifacts()

WINDOW_SIZE = 60      # Input: 5 menit sebelumnya (30 titik data)
FUTURE_STEPS = 60     # Output: 10 menit ke depan (60 titik data)

uploaded_file = st.file_uploader("📂 Upload File CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if 'ddate' not in df.columns or 'tag_value' not in df.columns:
            st.error("❌ CSV harus memiliki kolom 'ddate' dan 'tag_value'.")
        else:
            df['ddate'] = pd.to_datetime(df['ddate'])
            df = df.sort_values('ddate').reset_index(drop=True)

            st.subheader("📊 Data Terakhir:")
            st.dataframe(df.tail(5))

            if len(df) < WINDOW_SIZE:
                st.error(f"❌ Minimal {WINDOW_SIZE} baris diperlukan.")
            else:
                last_values = df['tag_value'].values[-WINDOW_SIZE:]
                scaled_input = scaler.transform(last_values.reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)

                forecast = []
                current_input = scaled_input.copy()

                for _ in range(FUTURE_STEPS):
                    pred = model.predict(current_input, verbose=0)[0, 0]
                    forecast.append(pred)
                    current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

                forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

                last_time = df['ddate'].iloc[-1]
                future_times = [last_time + timedelta(seconds=10 * (i + 1)) for i in range(FUTURE_STEPS)]

                result_df = pd.DataFrame({
                    'Datetime': future_times,
                    'Prediksi Tag Value': forecast_actual.flatten()
                })

                st.subheader("📈 Grafik Prediksi")
                st.line_chart(result_df.set_index("Datetime"))

                st.subheader("📋 Tabel Prediksi")
                st.dataframe(result_df)

                if len(df) >= WINDOW_SIZE + FUTURE_STEPS:
                    actual_future = df['tag_value'].values[-FUTURE_STEPS:]
                    mae = mean_absolute_error(actual_future, forecast_actual)
                    rmse = np.sqrt(mean_squared_error(actual_future, forecast_actual))

                    st.subheader("📉 Evaluasi Model (Data Uji)")
                    st.markdown(f"""
                    - **MAE**: {mae:.4f}  
                    - **RMSE**: {rmse:.4f}
                    """)
                else:
                    st.warning("⚠️ Data tidak cukup untuk evaluasi (butuh data aktual setelah input).")

    except Exception as e:
        st.error(f"❌ Error saat memproses data: {e}")
