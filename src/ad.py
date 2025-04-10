import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Fungsi untuk mengambil spektrum dari database
def spec(db_path, sample_name, iteration, lower_bound=None, upper_bound=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT wavelength, intensity
        FROM spectrum_data
        WHERE sample_name = ? AND iteration = ?
    """
    params = [sample_name, iteration]

    if lower_bound is not None and upper_bound is not None:
        query += " AND wavelength BETWEEN ? AND ?"
        params.extend([lower_bound, upper_bound])

    query += " ORDER BY wavelength"
    cursor.execute(query, params)
    data = cursor.fetchall()
    conn.close()

    if not data:
        print(f"No data found for sample: {sample_name}, iteration: {iteration}")
        return np.array([]), np.array([])

    wavelengths, intensities = zip(*data)
    return np.array(wavelengths), np.array(intensities)

# Fungsi untuk normalisasi dan PCA
def normalize_and_pca(spectra):
    combined_intensities = np.vstack([spectra[0][1], spectra[1][1], spectra[2][1]]).T

    # Apply PCA to the combined intensities
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(combined_intensities)

    # Normalize each iteration with Min-Max Scaling based on PC1
    scaler = MinMaxScaler()
    normalized_intensities = []
    for i in range(3):
        normalized_intensity = scaler.fit_transform(spectra[i][1].reshape(-1, 1)).flatten()
        normalized_intensities.append(normalized_intensity)

    return normalized_intensities

# Fungsi untuk menghitung rata-rata intensitas dari tiga iterasi
def average_normalized_spectra(wavelengths, normalized_intensities):
    avg_intensity = np.mean(normalized_intensities, axis=0)
    return wavelengths, avg_intensity

# Mengambil spektrum dari semua sampel, melakukan normalisasi, dan PCA
def get_and_process_spectra(db_path, sample_names, lower_bound=200, upper_bound=400):
    processed_spectra = {}
    for sample_name in sample_names:
        spectra = []
        for iteration in range(1, 4):  # Iterasi 1, 2, dan 3
            wavelengths, intensities = spec(db_path, sample_name, iteration, lower_bound, upper_bound)
            if len(wavelengths) > 0:
                spectra.append((wavelengths, intensities))

        # Apply normalization and PCA
        normalized_intensities = normalize_and_pca(spectra)

        # Calculate average normalized intensity
        wavelengths, avg_intensity = average_normalized_spectra(wavelengths, normalized_intensities)

        processed_spectra[sample_name] = (wavelengths, avg_intensity)
    return processed_spectra

# Path ke database dan nama-nama sampel
db_path = 'tanah_vulkanik.db'
sample_names = [f'S{i}' for i in range(4, 6)]  # Contoh: 2 sampel (S4 dan S5)

# Mengambil dan memproses spektrum dengan normalisasi dan PCA
processed_spectra = get_and_process_spectra(db_path, sample_names)

# Plot scatter dengan offset y untuk setiap sampel setelah normalisasi dan PCA
plt.figure(figsize=(10, 6))
offset = 1  # Offset untuk setiap sampel pada sumbu y
for i, (sample_name, (wavelengths, avg_intensity)) in enumerate(processed_spectra.items()):
    plt.plot(wavelengths, avg_intensity + i * offset, label=f'{sample_name} (y offset {i * offset:.1f})', linewidth=0.4)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Intensity (with y offset)')
plt.title('PCA and Min-Max Normalized Spectra with y-offset Scatter Plot')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('pca_normalized_spectra.pdf', format='pdf')

# Show the plot
plt.show()