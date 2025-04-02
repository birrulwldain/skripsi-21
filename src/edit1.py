import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_spectrum(spectrum):
    # Pastikan spektrum adalah array numpy
    spectrum = np.array(spectrum)

    # Geser spektrum agar nilai minimum menjadi 0
    min_val = np.min(spectrum)
    shifted_spectrum = spectrum - min_val

    # Normalisasi spektrum ke rentang [0, 1]
    max_val = np.max(shifted_spectrum)
    if max_val != 0:
        normalized_spectrum = shifted_spectrum / max_val
    else:
        normalized_spectrum = shifted_spectrum

    return normalized_spectrum


# Membaca file CSV dengan delimiter tab
def load_spectrum_from_csv(filename):
    """
    Load LIBS spectrum data from a tab-delimited CSV file.

    Parameters:
    - filename: string, path to the CSV file.

    Returns:
    - wavelengths: array-like, the wavelength values.
    - intensities: array-like, the intensity values (spectrum).
    """
    # Baca CSV dengan delimiter tab ('\t'), tampilkan 5 baris pertama untuk pengecekan
    data = pd.read_csv(filename, sep=',')
    print("Data CSV yang terbaca:")
    print(data.head())

    # Periksa apakah ada setidaknya 2 kolom
    if data.shape[1] < 2:
        raise ValueError("File CSV harus memiliki setidaknya dua kolom (panjang gelombang dan intensitas).")

    # Asumsi kolom pertama adalah panjang gelombang, dan kolom kedua adalah intensitas
    wavelengths = data.iloc[:, 0]
    intensities = data.iloc[:, 1]

    return wavelengths, intensities


# Contoh penggunaan
csv_filename = "tesasd.csv"  # Ganti dengan path file CSV Anda

try:
    # Memuat data spektrum dari CSV
    wavelengths, intensities = load_spectrum_from_csv(csv_filename)

    # Melakukan normalisasi spektrum
    normalized_spectrum = normalize_spectrum(intensities)

    # Plot hasil normalisasi
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, intensities, label="Original Spectrum")
    plt.plot(wavelengths, normalized_spectrum, label="Normalized Spectrum")
    plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Original vs Normalized LIBS Spectrum")
    plt.show()

    # Simpan spektrum yang dinormalisasi ke CSV baru
    output_filename = "normalized_spectrum.csv"
    normalized_data = pd.DataFrame({"Wavelength": wavelengths, "Normalized Intensity": normalized_spectrum})
    normalized_data.to_csv(output_filename, index=False)

except Exception as e:
    print(f"Terjadi kesalahan: {e}")