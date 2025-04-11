import os
import pandas as pd

# Path direktori tempat file CSV Anda berada
directory = '/Users/birrulwalidain/PycharmProjects/proyek-lIbs/'  # Ganti dengan path yang sesuai

# List untuk menyimpan data gabungan
combined_data = []

# Loop melalui semua file CSV di direktori
for filename in os.listdir(directory):
    if filename.endswith('_spectrum.csv'):
        # Buat path lengkap file CSV
        filepath = os.path.join(directory, filename)

        # Baca file CSV
        try:
            data = pd.read_csv(filepath)

            # Abaikan kolom pertama (tidak penting) dan gunakan kolom kedua dan ketiga
            if data.shape[1] >= 3:
                data = data.iloc[:, 1:3]  # Ambil kolom kedua (wavelength) dan ketiga (intensity)
                data.columns = ['wavelength', 'intensity']  # Beri nama kolom

                # Tambahkan kolom 'element' berdasarkan nama file (misalnya, 'Cu' dari 'Cu_spectrum.csv')
                element = filename.split('_')[0]
                data['element'] = element

                # Tambahkan data ini ke list
                combined_data.append(data)
            else:
                print(f"File {filename} tidak memiliki struktur yang sesuai.")
        except Exception as e:
            print(f"Error membaca file {filename}: {e}")

# Gabungkan semua data menjadi satu DataFrame jika ada data yang valid
if combined_data:
    all_spectra = pd.concat(combined_data, ignore_index=True)
    # Simpan DataFrame gabungan ke file CSV baru
    all_spectra.to_csv('combined_spectra.csv', index=False)
    print(f"Data berhasil disimpan ke 'combined_spectra.csv'.")
else:
    print("Tidak ada data yang dapat digabungkan.")