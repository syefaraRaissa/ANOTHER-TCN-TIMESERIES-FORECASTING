import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi 10 Menit ke Depan", layout="centered")

st.title("🔮 Prediksi Tag Value 10 Menit ke Depan")
st.markdown("Upload file CSV dengan kolom `TagValue`, sistem akan memprediksi nilai 10 menit ke depan berdasarkan 30 data terakhir.")

# Load model dan scaler
@st.cache_resource
def load_artifacts():
    model = load_model("my_model.h5", compile=False)
    scaler = joblib.load("scalercp.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Upload file CSV
uploaded_file = st.file_uploader("📂 Unggah file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'TagValue' not in df.columns:
        st.error("❌ Kolom 'TagValue' tidak ditemukan.")
    else:
        st.success("✅ File berhasil diunggah!")

        st.subheader("📊 Data Asli (10 Terakhir)")
        st.dataframe(df.tail(10))

        # Ambil 30 data terakhir
        data_window = df['TagValue'].values[-30:]
        
        if len(data_window) < 30:
            st.warning("⚠️ Data tidak cukup. Diperlukan minimal 30 titik data (misalnya 10 menit terakhir jika data per 10 detik).")
        else:
            # Skala data dan bentuk input
            scaled_input = scaler.transform(data_window.reshape(-1, 1))
            X_input = scaled_input.reshape(1, 30, 1)

            # Prediksi
            y_pred_scaled = model.predict(X_input)
            y_pred = scaler.inverse_transform(y_pred_scaled)[0][0]  # Ambil nilai float prediksi

            st.subheader("📈 Hasil Prediksi:")
            st.write(f"**Prediksi nilai 10 menit ke depan:** `{y_pred:.2f}`")

            # --- Evaluasi Model (jika data aktual tersedia)
            if len(df) >= 31:
                y_actual = df['TagValue'].values[-1]
                y_pred_arr = np.array([y_pred])
                y_true = np.array([y_actual])

                mae = mean_absolute_error(y_true, y_pred_arr)
                rmse = mean_squared_error(y_true, y_pred_arr, squared=False)

                st.subheader("🧪 Evaluasi Model:")
                st.write(f"**MAE (Mean Absolute Error):** {mae:.4f}")
                st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.4f}")
            else:
                st.info("📌 Data aktual untuk evaluasi belum tersedia (dibutuhkan nilai aktual setelah 10 menit).")

            # Grafik
            st.subheader("📉 Grafik Tag Value & Prediksi")
            fig, ax = plt.subplots()
            time = list(range(-29, 1))  # -29 s.d. 0 untuk 30 data historis
            ax.plot(time, data_window, label="Data Historis", marker='o')
            ax.plot([1], [y_pred], label="Prediksi (+10 Menit)", marker='x', color='red')
            ax.set_xlabel("Step (per 10 detik mundur)")
            ax.set_ylabel("TagValue")
            ax.set_title("TagValue Historis & Prediksi 10 Menit ke Depan")
            ax.legend()
            st.pyplot(fig)
