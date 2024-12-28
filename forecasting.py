import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

# Fungsi untuk memuat model
def load_model():
    return joblib.load("model_mlr.pkl")

def prediksi_harga_page():
    st.title("Prediksi Harga Rumah")

    # Load model
    model = load_model()

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
        "Neural Network"
    ])
    
    # Tombol Prediksi
    if st.button("Prediksi Harga"):
        # Periksa apakah semua input valid
        if LT > 0 and LB > 0:
            # Feature Engineering
            LT_LB = LT * LB
            LT_squared = LT ** 2
            LB_squared = LB ** 2
        # Format input menjadi array/dataframe sesuai dengan format model
            input_data = np.array([[
                float(LT),
                float(LB),
                float(JKT),
                float(JKM),
                float(GRS_encoded),
                float(LT_LB),
                float(LT_squared),
                float(LB_squared)
            ]])

            # Scalling data input
            input_scaled = scaler.transform(input_data)
            
            # Prediksi harga (log)
            log_harga_prediksi = model.predict(input_scaled)

            # Konversi kembali ke skala asli (inverse log)
            harga_prediksi = np.exp(log_harga_prediksi)

            
            # Tampilkan hasil prediksi
            st.success(f"Harga Prediksi: Rp {harga_prediksi[0]:,.2f}")
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
        x_train_scaled = scaler.transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)
        
        # Load model
        model = load_model()
        
        # Evaluasi pada data test
        y_test_pred = model.predict(x_test_scaled)
        r2 = r2_score(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
        st.write("Hasil Evaluasi:")
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"MAPE: {mape * 100:.2f}%")

        '''# Scatter Plot: Harga Aktual vs. Prediksi
        st.subheader("Harga Aktual vs. Prediksi")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_test_pred, ax=ax, alpha=0.7)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  
        ax.set_xlabel("Harga Aktual")
        ax.set_ylabel("Harga Prediksi")
        st.pyplot(fig)'''
    
    # Buat DataFrame untuk membandingkan hasil aktual dan prediksi
        comparison_df = pd.DataFrame({
        "Harga Aktual": np.expm1(y_test.values),  # Konversi log1p kembali ke nilai asli
        "Harga Prediksi": np.expm1(y_test_pred)  # Konversi log1p kembali ke nilai asli
    })

        # Tambahkan kolom selisih untuk analisis lebih lanjut
        comparison_df["Selisih"] = comparison_df["Harga Aktual"] - comparison_df["Harga Prediksi"]

        # Batasi ukuran tabel
        st.subheader("Perbandingan Harga Aktual dan Prediksi")
        st.dataframe(comparison_df.head(20), width=700)  # Menampilkan hanya 20 baris pertama 

        # Data metrik evaluasi
        metrics = {
            "Metrik": ["R² Score", "MSE", "RMSE", "MAE", "MAPE"],
            "Nilai": [r2, mse, rmse, mae, mape]
        }

        # Membuat DataFrame
        metrics_df = pd.DataFrame(metrics)

        # Membuat bar chart
        st.subheader("Grafik Perbandingan Metrik Evaluasi")
        fig, ax = plt.subplots()
        sns.barplot(x="Metrik", y="Nilai", data=metrics_df, ax=ax, palette="viridis")
        ax.set_title("Perbandingan Metrik Evaluasi")
        ax.set_ylabel("Nilai")
        st.pyplot(fig)

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
    1. **Multiple Linear Regression**: Cepat dan cocok untuk data dengan pola linier.  
    2. **Backpropagation Neural Network**: Memberikan prediksi yang lebih kompleks dengan akurasi yang lebih tinggi.  

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