import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

# Fungsi untuk memuat model
def load_model(algoritma):
    if algoritma == "Multiple Linear Regression":
        return joblib.load("model_mlr.pkl")
    elif algoritma == "Backpropagation Neural Network":
        return tf.keras.models.load_model("model_bpn.h5")

def prediksi_harga_page():
    st.title("Prediksi Harga Rumah")

    # Load scaler
    scaler = joblib.load('scaler.pkl')
    
    # Kolom Input
    col1, col2 = st.columns(2)
    
    with col1:
        LT = st.number_input("Luas Tanah (m²)", min_value=0, step=10)
        LB = st.number_input("Luas Bangunan (m²)", min_value=0, step=10)
        JKT = int(st.selectbox("Jumlah Kamar Tidur", [str(i) for i in range(1, 21)]))
    
    with col2:      
        JKM = int(st.selectbox("Jumlah Kamar Mandi", [str(i) for i in range(1, 21)]))   
        GRS = st.selectbox("Garasi", ["ADA", "TIDAK ADA"])
    
    # Map garasi ke nilai numerik 
    GRS_encoded = 1 if GRS == "ADA" else 0
       
    # Pilih Model Prediksi
    algoritma = st.radio("Pilih Algoritma Prediksi", [
        "Multiple Linear Regression", 
        "Backpropagation Neural Network"
    ])
    
    # Tombol Prediksi
    if st.button("Prediksi Harga"):
        # Periksa apakah semua input valid
        if LT > 0 and LB > 0:
            # Feature Engineering
            LT_LB = LT * LB
            LT_squared = LT ** 2
            LB_squared = LB ** 2
            LB_LT_Ratio = LB / LT if LT > 0 else 0
            LT_log = np.log1p(LT)
            LB_log = np.log1p(LB)
            
            # Format input menjadi array/dataframe sesuai dengan format model
            input_data = np.array([[
                float(LT),
                float(LB),
                float(JKT),
                float(JKM),
                float(GRS_encoded),
                float(LT_LB),
                float(LT_squared),
                float(LB_squared),
                float(LB_LT_Ratio),
                float(LT_log),
                float(LB_log)
            ]])

            # Scaling data input
            input_scaled = scaler.transform(input_data)
            
            # Load model sesuai algoritma
            model = load_model(algoritma)
            
            # Prediksi harga (log untuk kedua model)
            log_harga_prediksi = model.predict(input_scaled)
            
            
            # Konversi kembali ke skala asli (inverse log)
            harga_prediksi = np.exp(log_harga_prediksi)

            # Tampilkan hasil prediksi
            if algoritma == "Multiple Linear Regression":
                st.success(f"Harga Prediksi: Rp {harga_prediksi[0]:,.2f}")  
            else:  # Neural Network (BPN)
                st.success(f"Harga Prediksi: Rp {harga_prediksi[0][0]:,.2f}")  
        else:
            st.warning("Harap masukkan semua data dengan benar untuk melakukan prediksi.")

                   

def uji_akurasi_page():
    st.title("Uji Akurasi Model")
    
    uploaded_file = st.file_uploader("Upload Dataset untuk Evaluasi (.xlsx)", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df.drop(columns=['KOTA']) 
        df['GRS'] = df['GRS'].apply(lambda x: 1 if x == "ADA" else 0) 
        st.dataframe(df)

        # Penanganan Outliers (Metode IQR)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        cat_columns = ['GRS']
        numeric_columns = [col for col in numeric_columns if col not in cat_columns]

        Q1 = df[numeric_columns].quantile(0.25)
        Q3 = df[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1

        condition = ~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
        df_clean = df[condition]

        # Feature Engineering
        df_clean['LT_LB'] = df['LT'] * df['LB']
        df_clean['LT_squared'] = df['LT'] ** 2
        df_clean['LB_squared'] = df['LB'] ** 2
        df_clean['LB_LT_ratio'] = df_clean['LB'] / df_clean['LT']
        df_clean['LT_log'] = np.log1p(df_clean['LT'])
        df_clean['LB_log'] = np.log1p(df_clean['LB'])
            
        # Transformasi log target
        df_clean['HARGA'] = np.log1p(df['HARGA'])
        
        # Pisahkan x dan y
        y = df_clean['HARGA']
        x = df_clean.drop(columns=['HARGA'])

        # Bagi data menjadi train, val, dan test (70:20:10)
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.333, random_state=42)
        
        # Standarisasi data
        scaler = joblib.load('scaler.pkl')
        x_test_scaled = scaler.transform(x_test)
        
        # Load model MLR
        model_mlr = joblib.load('model_mlr.pkl')  
        
        # Evaluasi model MLR
        y_test_pred_mlr = model_mlr.predict(x_test_scaled)
        r2_mlr = r2_score(y_test, y_test_pred_mlr)
        mse_mlr = mean_squared_error(y_test, y_test_pred_mlr)
        rmse_mlr = np.sqrt(mse_mlr)
        mae_mlr = mean_absolute_error(y_test, y_test_pred_mlr)
        mape_mlr = mean_absolute_percentage_error(y_test, y_test_pred_mlr)

        # Load model BPN
        model_bpn = tf.keras.models.load_model('model_bpn.h5')  
        
        # Evaluasi model BPN
        y_test_pred_bpn = model_bpn.predict(x_test_scaled)
        r2_bpn = r2_score(y_test, y_test_pred_bpn)
        mse_bpn = mean_squared_error(y_test, y_test_pred_bpn)
        rmse_bpn = np.sqrt(mse_bpn)
        mae_bpn = mean_absolute_error(y_test, y_test_pred_bpn)
        mape_bpn = mean_absolute_percentage_error(y_test, y_test_pred_bpn)
        
        # Tampilkan hasil evaluasi
        st.title("Perbandingan Hasil Evaluasi Model:")

        # Kolom Input
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Multiple Linear Regression:")
            st.write(f"R² Score: {r2_mlr:.2f}")
            st.write(f"MSE: {mse_mlr:.2f}")
            st.write(f"RMSE: {rmse_mlr:.2f}")
            st.write(f"MAE: {mae_mlr:.2f}")
            st.write(f"MAPE: {mape_mlr * 100:.2f}%")

        with col2:
            st.write("#### Backpropagation Neural Network:")
            st.write(f"R² Score: {r2_bpn:.2f}")
            st.write(f"MSE: {mse_bpn:.2f}")
            st.write(f"RMSE: {rmse_bpn:.2f}")
            st.write(f"MAE: {mae_bpn:.2f}")
            st.write(f"MAPE: {mape_bpn * 100:.2f}%")

        # Scatter Plot Perbandingan
        st.title("Scatter Plot Harga Aktual vs Harga Prediksi:")
        plt.figure(figsize=(8, 4))

        # Transform balik (exp) untuk MLR
        actual_prices = np.exp(y_test) - 1  # karena kita menggunakan log1p sebelumnya
        mlr_predicted_prices = np.exp(y_test_pred_mlr.flatten()) - 1
        bpn_predicted_prices = np.exp(y_test_pred_bpn.flatten()) - 1

        # Scatter plot untuk MLR
        plt.scatter(actual_prices, mlr_predicted_prices, 
                color='blue', alpha=0.6, label='MLR', s=50)

        # Scatter plot untuk BPN
        plt.scatter(actual_prices, bpn_predicted_prices, 
                color='green', alpha=0.6, label='BPN', s=50)

        # Garis prediksi ideal
        min_val = min(actual_prices.min(), mlr_predicted_prices.min(), bpn_predicted_prices.min())
        max_val = max(actual_prices.max(), mlr_predicted_prices.max(), bpn_predicted_prices.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                color='red', linestyle='--', label='Ideal Prediction')

        # Label dan judul
        plt.title("Harga Aktual vs Harga Prediksi", fontsize=12, pad=20)
        plt.xlabel("Harga Aktual", fontsize=10)
        plt.ylabel("Harga Prediksi", fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout()

        # Tampilkan plot di Streamlit
        st.pyplot(plt)

    

def tentang_aplikasi_page():
    st.title("Tentang Aplikasi")
    st.write("""
    Selamat datang di **Aplikasi Real Estate Prediction**!  

    Aplikasi ini dirancang untuk membantu Anda memprediksi **harga rumah** berdasarkan berbagai variabel penting, seperti:  
    - **Luas Tanah**  
    - **Luas Bangunan**  
    - **Jumlah Kamar Tidur**  
    - **Jumlah Kamar Mandi**  
    - **Keberadaan Garasi**  

    Dengan memanfaatkan teknologi **Machine Learning**, aplikasi ini menggunakan dua algoritma utama:  
    1. **Multiple Linear Regression**
    2. **Backpropagation Neural Network**

    Aplikasi **Real Estate Prediction** ini dirancang tidak hanya untuk memberikan hasil prediksi yang akurat, tetapi juga untuk membantu pengguna dalam memahami faktor-faktor yang memengaruhi harga rumah.   
    """)
   

def main():
    # Konfigurasi Halaman
    st.set_page_config(
        page_title="Prediksi Harga Rumah", 
        page_icon=":house:",
        layout="wide"
    )
    
    # Sidebar Menu Vertical
    with st.sidebar:
        selected = option_menu(
            menu_title="Real Estate Prediction",
            options=["Prediksi Harga", "Uji Akurasi", "Tentang Aplikasi"],
            icons=["calculator", "graph-up", "info-circle"],
            menu_icon="house",
            default_index=0
        )
    
    # Routing Halaman
    if selected == "Prediksi Harga":
        prediksi_harga_page()
    elif selected == "Uji Akurasi":
        uji_akurasi_page()
    elif selected == "Tentang Aplikasi":
        tentang_aplikasi_page()

# Jalankan Aplikasi
if __name__ == "__main__":
    main()