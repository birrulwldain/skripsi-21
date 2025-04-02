import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from scipy.integrate import simpson  # Untuk melakukan integrasi numerik

# Fungsi untuk mengambil data dari database berdasarkan iterasi
def get_spectrum_data(db_path, sample_name, iteration):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT wavelength, intensity
        FROM spectrum_data
        WHERE sample_name = ? AND iteration = ?
        ORDER BY wavelength
    """
    cursor.execute(query, (sample_name, iteration))
    data = cursor.fetchall()
    conn.close()

    if data:
        wavelengths, intensities = zip(*data)
        return np.array(wavelengths), np.array(intensities)
    else:
        print(f"Tidak ada data ditemukan untuk sampel: {sample_name}, iterasi: {iteration}")
        return np.array([]), np.array([])

# Fungsi untuk menampilkan label puncak yang melebihi threshold tertentu
def label_peaks(ax, wavelengths, intensities, threshold):
    for i, intensity in enumerate(intensities):
        if intensity > threshold:
            ax.annotate(f'{wavelengths[i]:.6f} nm',
                        xy=(wavelengths[i], intensity),
                        xytext=(wavelengths[i], intensity + 0.05*intensity),
                        arrowprops=dict(arrowstyle="->", color='red'),
                        fontsize=8, color='red')

# Fungsi untuk merata-ratakan spektrum berdasarkan luas kurva
def average_spectrum_area(db_path, sample_name, iterations, threshold=1e5):
    all_wavelengths = []
    all_intensities = []
    integrated_areas = []

    for iteration in iterations:
        wavelengths, intensities = get_spectrum_data(db_path, sample_name, iteration)

        if len(wavelengths) == 0 or len(intensities) == 0:
            continue

        all_wavelengths.append(wavelengths)
        all_intensities.append(intensities)

        # Plot untuk setiap iterasi tanpa interpolasi
        plt.figure()
        plt.plot(wavelengths, intensities, label=f'Iteration {iteration}', marker='o', linestyle='--')
        plt.grid(True)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title(f'Spectrum for {sample_name} - Iteration {iteration}')
        label_peaks(plt.gca(), wavelengths, intensities, threshold)
        plt.legend()
        plt.show()

    if not all_intensities:
        print(f"Tidak ada data yang bisa diproses untuk sampel: {sample_name}")
        return

    # Cari rentang panjang gelombang umum
    min_wavelength = max(w.min() for w in all_wavelengths)
    max_wavelength = min(w.max() for w in all_wavelengths)

    # Buat grid panjang gelombang yang sama untuk semua spektrum
    common_wavelengths = np.linspace(min_wavelength, max_wavelength, num=1000)

    # Interpolasi linier pada semua spektrum untuk menyamakan panjang gelombang
    interpolated_intensities = []
    for wavelengths, intensities in zip(all_wavelengths, all_intensities):
        interpolated_intensity = np.interp(common_wavelengths, wavelengths, intensities)
        interpolated_intensities.append(interpolated_intensity)

        # Hitung luas kurva menggunakan metode trapezoidal (integrasi numerik)
        area = simpson(interpolated_intensity, common_wavelengths)
        integrated_areas.append(area)

    # Hitung rata-rata luas kurva dari semua iterasi
    avg_area = np.mean(integrated_areas)

    # Plot spektrum rata-rata
    avg_intensities = np.mean(interpolated_intensities, axis=0)
    plt.figure()
    plt.plot(common_wavelengths, avg_intensities, label=f'Average Spectrum (Area: {avg_area:.2f})', color='blue')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title(f'Average Spectrum for {sample_name} (based on area)')
    label_peaks(plt.gca(), common_wavelengths, avg_intensities, threshold)
    plt.legend()
    plt.grid(True)
    plt.show()

    return common_wavelengths, avg_intensities, avg_area

# Main function to run the averaging
def main():
    db_path = "tanah_vulkanik.db"  # Path ke database
    sample_name = "S7"  # Nama sampel
    iterations = [1, 2, 3]  # Iterasi yang akan digunakan

    # Lakukan perataan spektrum berdasarkan luas kurva
    common_wavelengths, avg_intensities, avg_area = average_spectrum_area(db_path, sample_name, iterations)
    print(f"Luas kurva rata-rata dari spektrum: {avg_area:.2f}")

if __name__ == "__main__":
    main()