import requests
import os

# Konfigurasi
REPO = "birrulwldain/skripsi-21"  # Ganti dengan repo kamu
TOKEN = os.getenv("GH_TOKEN")  # Set token via environment variable

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json"
}

issues = [
    # === 1. Data Preparation ===
    {
        "title": "[DP1] Koleksi Data Spektrum LIBS",
        "body": "Mengumpulkan dataset spektrum emisi (eksperimen atau publik).\n\n**Acceptance Criteria:**\n- Data tersimpan dalam format CSV/HDF5\n- Kolom: panjang gelombang, intensitas, label unsur.",
        "labels": ["data-prep", "high"],
    },
    {
        "title": "[DP2] Normalisasi Data",
        "body": "Normalisasi intensitas spektrum (Min-Max/Z-Score).\n\n**Acceptance Criteria:**\n- Data terstandarisasi dengan rentang [0,1] atau distribusi normal.",
        "labels": ["data-prep", "high"],
    },
    {
        "title": "[DP3] Segmentasi Dataset",
        "body": "Bagi data menjadi training, validation, dan test set (70-15-15).\n\n**Acceptance Criteria:**\n- Pembagian data terdokumentasi dan terpisah.",
        "labels": ["data-prep", "high"],
    },
    {
        "title": "[DP4] Augmentasi Data",
        "body": "Tambahkan noise atau geser spektrum untuk augmentasi.\n\n**Acceptance Criteria:**\n- Dataset diperbesar 2x lipat dengan variasi realistis.",
        "labels": ["data-prep", "medium"],
    },

    # === 2. Model Development ===
    {
        "title": "[MD1] Desain Arsitektur LSTM",
        "body": "Definisikan layer input, LSTM, dropout, dense.\n\n**Acceptance Criteria:**\n- Arsitektur model terdokumentasi dengan diagram.",
        "labels": ["model", "high"],
    },
    {
        "title": "[MD2] Implementasi Bidirectional LSTM",
        "body": "Tambahkan lapisan bidirectional untuk konteks maju-mundur.\n\n**Acceptance Criteria:**\n- Model mampu memproses data dua arah.",
        "labels": ["model", "medium"],
    },
    {
        "title": "[MD3] Implementasi Dropout",
        "body": "Tambahkan dropout layer untuk mengurangi overfitting.\n\n**Acceptance Criteria:**\n- Dropout rate 0.3–0.5.",
        "labels": ["model", "high"],
    },
    {
        "title": "[MD4] Konfigurasi Loss & Optimizer",
        "body": "Pilih loss function (MSE/Cross-Entropy) dan optimizer (Adam).\n\n**Acceptance Criteria:**\n- Model dapat dikompilasi tanpa error.",
        "labels": ["model", "high"],
    },

    # === 3. Training & Optimization ===
    {
        "title": "[TO1] Pelatihan Model",
        "body": "Latih model dengan data training.\n\n**Acceptance Criteria:**\n- Loss training menurun dalam 20 epoch.",
        "labels": ["training", "high"],
    },
    {
        "title": "[TO2] Early Stopping",
        "body": "Implementasi callback untuk hentikan pelatihan jika stagnan.\n\n**Acceptance Criteria:**\n- Pelatihan berhenti jika val_loss tidak membaik dalam 10 epoch.",
        "labels": ["training", "medium"],
    },
    {
        "title": "[TO3] Hiperparameter Tuning",
        "body": "Optimasi unit LSTM, dropout rate, learning rate.\n\n**Acceptance Criteria:**\n- Model dengan val_loss terbaik terseleksi.",
        "labels": ["training", "medium"],
    },
    {
        "title": "[TO4] Simpan Model Terbaik",
        "body": "Simpan model dalam format .h5 atau .pb.\n\n**Acceptance Criteria:**\n- Model tersimpan dan dapat di-load ulang.",
        "labels": ["training", "low"],
    },

    # === 4. Evaluasi & Validasi ===
    {
        "title": "[EV1] Evaluasi Metrik Regresi/Klasifikasi",
        "body": "Hitung MSE, RMSE, R² (regresi) atau accuracy, F1-score (klasifikasi).\n\n**Acceptance Criteria:**\n- Laporan metrik lengkap.",
        "labels": ["evaluation", "high"],
    },
    {
        "title": "[EV2] Visualisasi Prediksi vs Aktual",
        "body": "Plot hasil prediksi vs data aktual.\n\n**Acceptance Criteria:**\n- Grafik visual jelas dan mudah dipahami.",
        "labels": ["evaluation", "medium"],
    },
    {
        "title": "[EV3] Uji Model pada Data Noisy",
        "body": "Evaluasi model dengan data gangguan (noise, baseline drift).\n\n**Acceptance Criteria:**\n- Model toleran terhadap gangguan ringan.",
        "labels": ["evaluation", "medium"],
    },
    {
        "title": "[EV4] Validasi dengan Metode Konvensional",
        "body": "Bandingkan hasil LSTM dengan metode Beer-Lambert/CF-LIBS.\n\n**Acceptance Criteria:**\n- Laporan perbandingan akurasi.",
        "labels": ["evaluation", "high"],
    },

    # === 5. Integrasi dengan Sistem LIBS ===
    {
        "title": "[IN1] Pipeline Prediksi Real-Time",
        "body": "Integrasi model dengan alat LIBS untuk prediksi real-time.\n\n**Acceptance Criteria:**\n- Prediksi dapat dijalankan dalam waktu <1 detik per spektrum.",
        "labels": ["integration", "low"],
    },
    {
        "title": "[IN2] Kalibrasi Model dengan Data Eksperimen",
        "body": "Sesuaikan model dengan data eksperimen aktual.\n\n**Acceptance Criteria:**\n- Model dapat memprediksi konsentrasi unsur dengan error <10%.",
        "labels": ["integration", "high"],
    },

    # === 6. Dokumentasi & Deployment ===
    {
        "title": "[DD1] Dokumentasi Kode",
        "body": "Tulis dokumentasi fungsi dan workflow.\n\n**Acceptance Criteria:**\n- Dokumentasi tersedia dalam format README.md.",
        "labels": ["documentation", "medium"],
    },
    {
        "title": "[DD2] Docker Container",
        "body": "Package model dalam container untuk deployment.\n\n**Acceptance Criteria:**\n- Container dapat dijalankan di Ubuntu/Windows.",
        "labels": ["deployment", "low"],
    },
    {
        "title": "[DD3] API untuk Prediksi",
        "body": "Bangun REST API menggunakan Flask/FastAPI.\n\n**Acceptance Criteria:**\n- API dapat menerima input spektrum dan return prediksi.",
        "labels": ["deployment", "low"],
    },
]



for issue in issues:
    url = f"https://api.github.com/repos/{REPO}/issues"
    res = requests.post(url, headers=headers, json=issue)
    print(f"Issue '{issue['title']}': {res.status_code}")
