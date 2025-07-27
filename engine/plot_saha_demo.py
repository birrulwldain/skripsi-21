import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Tambahkan path engine ke sys.path agar bisa mengimpor simul1
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    from simul1 import DataFetcher, SpectrumSimulator, MixedSpectrumSimulator, SIMULATION_CONFIG, ionization_energies
except ImportError as e:
    print(f"Gagal mengimpor dari simul1.py: {e}")
    print("Pastikan skrip ini dijalankan dari direktori 'engine' atau path-nya benar.")
    sys.exit(1)

def setup_atomic_data(atomic_data_path):
    """Memuat data energi ionisasi dari file HDF5."""
    import h5py
    import pandas as pd
    global ionization_energies
    try:
        with h5py.File(atomic_data_path, 'r') as f:
            dset = f['elements']
            columns = dset.attrs['columns']
            df_ionization = pd.DataFrame(
                [[item[0], item[1].decode('utf-8'), item[2].decode('utf-8'), item[3].decode('utf-8'),
                  item[4].decode('utf-8'), item[5], item[6].decode('utf-8')] for item in dset[:]],
                columns=columns
            )
            for _, row in df_ionization.iterrows():
                ionization_energies[row["Sp. Name"]] = float(row["Ionization Energy (eV)"])
    except Exception as e:
        print(f"Error saat membaca file data atom: {e}")
        sys.exit(1)

def get_strongest_lines(element, ion_state, nist_path, num_lines=5, wl_range=(250, 450)):
    """Mengambil beberapa garis spektral terkuat untuk anotasi."""
    fetcher = DataFetcher(nist_path)
    nist_data, _ = fetcher.get_nist_data(element, ion_state)
    if not nist_data:
        return []
    
    # Filter berdasarkan rentang panjang gelombang
    strong_lines = [line for line in nist_data if wl_range[0] <= float(line[0]) <= wl_range[1]]
    
    # Urutkan berdasarkan Aki (probabilitas transisi) untuk menemukan yang terkuat
    strong_lines.sort(key=lambda x: float(x[1]), reverse=True)
    
    # Ambil panjang gelombang dari garis terkuat
    return [float(line[0]) for line in strong_lines[:num_lines]]

def generate_single_element_spectrum(element, temperature, electron_density, nist_path):
    """Menghasilkan spektrum untuk satu elemen pada kondisi tertentu."""
    fetcher = DataFetcher(nist_path)
    simulators = []
    for ion in [1, 2]:
        nist_data, _ = fetcher.get_nist_data(element, ion)
        if nist_data:
            ion_energy = ionization_energies.get(f"{element} {'I' if ion == 1 else 'II'}", 0.0)
            simulators.append(SpectrumSimulator(nist_data, element, ion, temperature, ion_energy))

    if not simulators:
        print(f"Tidak ada data untuk {element}.")
        return None, None, None

    mixer = MixedSpectrumSimulator(simulators, electron_density, fetcher.delta_E_max)
    wavelengths, spectrum, params = mixer.generate_spectrum([(element, 100.0)], temperature)
    return wavelengths, spectrum, params

def main():
    """Fungsi utama untuk membuat dan menyimpan plot."""
    temperatures = [6000, 30000]
    element = 'Al'
    electron_density = 1e17
    plot_wl_range = (250, 450)
    
    # Tentukan path file data
    nist_data_file = os.path.join(script_dir, "data", "nist_data(1).h5")
    atomic_data_file = os.path.join(script_dir, "data", "atomic_data1.h5")
    
    # Tentukan direktori output
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    output_dir = os.path.join(base_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    print("Memuat data atomik...")
    setup_atomic_data(atomic_data_file)

    print("Mengambil data garis spektral terkuat untuk anotasi...")
    al_i_lines = get_strongest_lines('Al', 1, nist_data_file, num_lines=5, wl_range=plot_wl_range)
    al_ii_lines = get_strongest_lines('Al', 2, nist_data_file, num_lines=5, wl_range=plot_wl_range)
    print(f"  - Garis Al I ditemukan: {al_i_lines}")
    print(f"  - Garis Al II ditemukan: {al_ii_lines}")

    # Kumpulkan semua spektrum untuk menentukan batas sumbu-y global
    all_spectra = []
    print("Menghasilkan spektrum untuk setiap suhu...")
    for temp in temperatures:
        print(f"  - Mensimulasikan Al pada {temp} K...")
        _, spectrum, _ = generate_single_element_spectrum(element, temp, electron_density, nist_data_file)
        if spectrum is not None:
            all_spectra.append(spectrum)

    if not all_spectra:
        print("Gagal menghasilkan spektrum. Proses dihentikan.")
        return

    global_max_intensity = max(np.max(s) for s in all_spectra) * 1.15 # Beri sedikit ruang lebih untuk anotasi

    # Buat dan simpan plot untuk setiap suhu
    print("Membuat dan menyimpan plot...")
    for i, temp in enumerate(temperatures):
        wavelengths, spectrum, params = generate_single_element_spectrum(element, temp, electron_density, nist_data_file)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(wavelengths, spectrum, color='navy')
        
        ax.set_xlabel('Panjang Gelombang (nm)', fontsize=12)
        ax.set_ylabel('Intensitas Relatif (a.u.)', fontsize=12)
        
        ax.set_ylim(0, global_max_intensity)
        ax.set_xlim(plot_wl_range)
        
        ax.grid(True, linestyle='--', alpha=0.6)

        # Tambahkan anotasi
        for wl in al_i_lines:
            idx = np.argmin(np.abs(wavelengths - wl))
            y = spectrum[idx]
            if y > 0.01 * global_max_intensity: # Hanya anotasi puncak yang terlihat
                ax.annotate('Al I', xy=(wl, y), xytext=(wl, y + 0.05 * global_max_intensity),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                            ha='center', fontsize=9, color='blue')

        for wl in al_ii_lines:
            idx = np.argmin(np.abs(wavelengths - wl))
            y = spectrum[idx]
            if y > 0.01 * global_max_intensity: # Hanya anotasi puncak yang terlihat
                ax.annotate('Al II', xy=(wl, y), xytext=(wl, y + 0.05 * global_max_intensity),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                            ha='center', fontsize=9, color='red')
        
        # Simpan plot
        filename = f"al_spectrum_{temp}K.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"  - Plot disimpan di: {save_path}")

    print("\nSemua plot telah berhasil dibuat.")

if __name__ == "__main__":
    main()