import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="HeartGuard AI: Ensemble",
    page_icon="🫀",
    layout="wide"
)

# --- 2. LOAD MODEL & ASET ---
@st.cache_resource
def load_assets():
    try:
        # Pastikan file-file ini ada di folder yang sama
        model = joblib.load('heart_attack_rf_model.pkl') # Model Ensemble kamu
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except Exception as e:
        return None, None, None

model, scaler, feature_names = load_assets()

# --- 3. SIDEBAR: PROFIL PASIEN ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063823.png", width=80)
    st.title("Profil Pasien")
    
    age = st.slider("Usia", 20, 90, 45)
    gender = st.radio("Jenis Kelamin", ["Male", "Female"])
    region = st.selectbox("Wilayah", ["Urban (Kota)", "Rural (Desa)"])
    income = st.select_slider("Ekonomi", ["Low", "Middle", "High"], value="Middle")

# --- 4. HEADER ---
st.title("🫀 HeartGuard AI: Ensemble Diagnostic")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <strong>Status Model:</strong> Ensemble Learning (XGBoost + RF + GB) <br>
    Model ini menggunakan perhitungan medis (Tekanan Arteri Rata-rata, Rasio Kolesterol) untuk prediksi yang objektif.
</div>
""", unsafe_allow_html=True)

# --- 5. INPUT DATA (TABS) ---
tab1, tab2, tab3 = st.tabs(["🏥 Data Klinis (Lab)", "🏃 Gaya Hidup", "📋 Riwayat Kesehatan"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tanda Vital")
        sys_bp = st.number_input("Tensi Sistolik (mmHg)", 90, 250, 120, help="Angka atas tekanan darah")
        dia_bp = st.number_input("Tensi Diastolik (mmHg)", 50, 150, 80, help="Angka bawah tekanan darah")
        ekg = st.selectbox("Hasil EKG", ["Normal", "Abnormal"])
        
    with col2:
        st.subheader("Profil Lemak & Gula")
        chol = st.number_input("Kolesterol Total", 100, 500, 200)
        ldl = st.number_input("LDL (Lemak Jahat)", 50, 300, 100)
        hdl = st.number_input("HDL (Lemak Baik)", 20, 100, 50)
        sugar = st.number_input("Gula Darah Puasa", 70, 400, 100)
        trig = st.number_input("Trigliserida", 50, 500, 150)

with tab2:
    col_life1, col_life2 = st.columns(2)
    with col_life1:
        smoking = st.selectbox("Status Merokok", ["Never", "Past", "Current"])
        alcohol = st.selectbox("Konsumsi Alkohol", ["None", "Moderate", "High"])
        diet = st.radio("Pola Makan", ["Healthy", "Unhealthy"], horizontal=True)
        
    with col_life2:
        activity = st.select_slider("Aktivitas Fisik", ["Low", "Moderate", "High"])
        stress = st.select_slider("Tingkat Stres", ["Low", "Moderate", "High"])
        pollution = st.select_slider("Paparan Polusi", ["Low", "Moderate", "High"])
        sleep = st.slider("Jam Tidur", 3, 12, 7)

with tab3:
    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        has_diabetes = st.checkbox("Riwayat Diagnosis Diabetes?")
        fam_hist = st.checkbox("Riwayat Keluarga Sakit Jantung?")
    with col_hist2:
        prev_heart = st.checkbox("Pernah Serangan Jantung Sebelumnya?")
        meds = st.checkbox("Rutin Minum Obat Jantung?")
        waist = st.number_input("Lingkar Pinggang (cm)", 50, 150, 80)

# --- 6. LOGIKA PREDIKSI ---
if st.button("🔍 ANALISIS RISIKO", type="primary", use_container_width=True):
    if model is None:
        st.error("⚠️ File model (.pkl) tidak ditemukan. Pastikan sudah di-upload.")
    else:
        # A. Mapping Data Kategorikal (Sesuai Notebook)
        maps = {
            'income_level': {'Low': 0, 'Middle': 1, 'High': 2},
            'stress_level': {'Low': 0, 'Moderate': 1, 'High': 2},
            'physical_activity': {'Low': 0, 'Moderate': 1, 'High': 2},
            'air_pollution_exposure': {'Low': 0, 'Moderate': 1, 'High': 2},
            'alcohol_consumption': {'None': 0, 'Moderate': 1, 'High': 2},
            'smoking_status': {'Never': 0, 'Past': 1, 'Current': 2},
            'region': {'Rural (Desa)': 0, 'Urban (Kota)': 1},
            'gender': {'Female': 0, 'Male': 1}
        }
        
        # B. Siapkan DataFrame Mentah
        input_data = {
            'age': age,
            'gender': maps['gender'][gender],
            'region': maps['region'][region],
            'income_level': maps['income_level'][income],
            'smoking_status': maps['smoking_status'][smoking],
            'alcohol_consumption': maps['alcohol_consumption'][alcohol],
            'physical_activity': maps['physical_activity'][activity],
            'dietary_habits': 1 if diet == 'Unhealthy' else 0,
            'air_pollution_exposure': maps['air_pollution_exposure'][pollution],
            'stress_level': maps['stress_level'][stress],
            'sleep_hours': sleep,
            'blood_pressure_systolic': sys_bp,
            'blood_pressure_diastolic': dia_bp,
            'cholesterol_level': chol,
            'cholesterol_ldl': ldl,
            'cholesterol_hdl': hdl,
            'triglycerides': trig,
            'fasting_blood_sugar': sugar,
            'waist_circumference': waist,
            'EKG_results': 1 if ekg == 'Abnormal' else 0,
            'diabetes': 1 if has_diabetes else 0, # Input manual atau dari gula darah
            'family_history': int(fam_hist),
            'previous_heart_disease': int(prev_heart),
            'medication_usage': int(meds),
            # Fitur dummy untuk obesity/screening jika diperlukan oleh scaler lama
            'obesity': 0, 
            'participated_in_free_screening': 0
        }
        
        df_input = pd.DataFrame([input_data])
        
        # C. FEATURE ENGINEERING (BAGIAN PENTING)
        # 1. Pulse Pressure & MAP
        df_input['pulse_pressure'] = df_input['blood_pressure_systolic'] - df_input['blood_pressure_diastolic']
        df_input['map'] = df_input['blood_pressure_diastolic'] + (df_input['pulse_pressure'] / 3)
        
        # 2. Rasio Kolesterol
        df_input['cholesterol_ratio'] = df_input['cholesterol_level'] / (df_input['cholesterol_hdl'] + 1)
        df_input['ldl_hdl_ratio'] = df_input['cholesterol_ldl'] / (df_input['cholesterol_hdl'] + 1)
        
        # 3. Interaksi Risiko & PERBAIKAN NAMA KOLOM
        # Ini baris baru yang WAJIB ada biar error hilang:
        df_input['hypertension'] = (df_input['blood_pressure_systolic'] > 140).astype(int)  # <--- TAMBAHKAN INI
        
        # Kolom engineer lainnya
        df_input['is_hypertension'] = (df_input['blood_pressure_systolic'] > 140).astype(int)
        df_input['is_diabetes'] = (df_input['diabetes'] == 1).astype(int)
        df_input['risk_score_simple'] = df_input['smoking_status'] + df_input['is_hypertension'] + df_input['is_diabetes'] + df_input['dietary_habits']
        # D. Filter Kolom (Hanya ambil yg dikenali model)
        if feature_names:
            # Ambil irisan kolom yang ada di input dan feature_names
            final_cols = [c for c in feature_names if c in df_input.columns]
            df_input = df_input[final_cols]
            
        try:
            # E. Scaling & Prediksi
            X_scaled = scaler.transform(df_input)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            
            # F. Tampilan Hasil
            st.divider()
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("Probabilitas Risiko", f"{probability*100:.2f}%")
                if prediction == 1:
                    st.error("⚠️ PREDIKSI: BERISIKO")
                else:
                    st.success("✅ PREDIKSI: AMAN")
            
            with col_res2:
                st.write("### 📝 Analisis Medis Otomatis:")
                reasons = []
                if sys_bp > 140: reasons.append(f"- **Hipertensi:** Tensi sistolik {sys_bp} mmHg terlalu tinggi.")
                if chol > 200: reasons.append(f"- **Kolesterol:** Total {chol} mg/dL di atas batas normal.")
                if df_input['ldl_hdl_ratio'].values[0] > 3.0: reasons.append("- **Rasio Lemak Buruk:** Perbandingan LDL/HDL tidak seimbang.")
                if smoking == "Current": reasons.append("- **Merokok:** Faktor risiko utama kardiovaskular.")
                
                if reasons:
                    for r in reasons:
                        st.write(r)
                else:
                    st.write("- Semua parameter vital terlihat dalam batas wajar.")
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan teknis: {str(e)}")
            st.warning("Pastikan file 'scaler.pkl' dan 'feature_names.pkl' adalah versi terbaru dari training Ensemble.")