# 🏦 Bank DSS — Analytical Command Center
**Decision Support System untuk Prediksi Kampanye Telemarketing Bank**

> Tema: Analytical Command Center | Warna: Purple-Teal  
> Referensi: Moro, S., Cortez, P., & Rita, P. (2014). Decision Support Systems, Elsevier, 62, 22–31.

---

## 🚀 Quick Start (2 langkah)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup data & model (WAJIB sekali di awal)
```bash
python3 setup_models.py
```

### 3. Jalankan aplikasi
```bash
streamlit run app.py
```

---

## 📁 Struktur Proyek
```
dss_bank/
├── app.py                  # ← Aplikasi utama Streamlit
├── setup_models.py         # ← Script training model (run sekali)
├── requirements.txt
├── data/
│   └── bank_data.csv       # ← Generated oleh setup_models.py
└── models/
    ├── lr_model.pkl        # Logistic Regression
    ├── rf_model.pkl        # Random Forest
    ├── kmeans.pkl          # K-Means Clustering
    ├── scaler.pkl          # StandardScaler
    ├── encoders.pkl        # Label Encoders
    └── ...
```

---

## 📊 6 Modul DSS

| Halaman | Modul | Kelompok | Teori |
|---------|-------|----------|-------|
| Overview & Data Explorer | Data Pipeline | 1,2,3 | Intelligence Phase |
| Segmentasi Customer | K-Means (k=4) | 1,4 | STP Framework |
| Prediksi Respons | LR + RF Ensemble | 2,4 | Choice Phase |
| Analisis Efektivitas | Statistical Aggregation | 1,2,3 | Design Phase |
| Simulasi Budget | Monte Carlo | 6 | Risk Analysis |
| Rekomendasi Keputusan | MCDM + Rule Engine | 7 | Simon (1977) |

---

## 💡 Untuk Production (Ganti Dataset Asli)

1. Download `bank-additional-full.csv` dari UCI Repository
2. Ganti di `setup_models.py`:
   ```python
   # Ganti baris generate synthetic data dengan:
   df = pd.read_csv('data/bank-additional-full.csv', sep=';')
   df['y'] = (df['y'] == 'yes').astype(int)
   ```
3. Train di Google Colab (RAM lebih besar)
4. Copy file `.pkl` hasil training ke folder `models/`
5. Streamlit hanya **load & infer** — tidak retrain

---

## ⚙️ Optimasi RAM 4GB

- `@st.cache_data` untuk dataset
- `@st.cache_resource` untuk models
- Load model per-page (lazy loading)
- Random Forest: max 50 trees, depth 8
- Tidak menggunakan XGBoost atau neural network

---

## 📖 Referensi Teori

**Simon's Decision Model (1977):**
- **Intelligence**: Mengidentifikasi masalah (Overview + Segmentasi)
- **Design**: Membangun alternatif (Prediksi + Analisis)
- **Choice**: Memilih keputusan terbaik (Simulasi + Rekomendasi)

**Dataset**: UCI Bank Marketing Dataset  
Moro, S., Cortez, P., & Rita, P. (2014). DOI: 10.1016/j.dss.2014.03.001
