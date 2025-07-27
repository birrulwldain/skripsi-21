import os
import sys
import pandas as pd
import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Mengimpor semua kelas yang diperlukan dari simul1.py
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(script_dir) 
sys.path.append(engine_path)

try:
    from simul1 import DataFetcher, SpectrumSimulator, MixedSpectrumSimulator, ionization_energies
except ImportError:
    print("Gagal mengimpor dari 'simul1'. Pastikan file simul1.py berada di path yang benar.")
    sys.exit(1)

def setup_atomic_data(atomic_data_path):
    """Memuat data energi ionisasi dari file HDF5."""
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
        return True
    except Exception as e:
        print(f"Error saat membaca file data atom: {e}")
        return False

# ---------------------------------------------------------------------------
# Fungsi Utama Analisis (Dengan Perbaikan)
# ---------------------------------------------------------------------------
def perform_temperature_study(elements, temperatures, electron_density, nist_path, num_lines_per_state=2):
    """Melakukan studi parametrik pengaruh suhu terhadap intensitas garis spektral."""
    
    fetcher = DataFetcher(nist_path)
    mixed_sim = MixedSpectrumSimulator([], electron_density, {})
    final_results = []

    for elem in elements:
        for temp in temperatures:
            print(f"Menganalisis {elem} pada {temp} K...")
            ion_energy = ionization_energies.get(f"{elem} I")
            if not ion_energy:
                print(f"Peringatan: Energi ionisasi untuk {elem} I tidak ditemukan. Melewati...")
                continue
            saha_ratio = mixed_sim.saha_ratio(ion_energy, temp, electron_density * 1e6)
            ion_fraction = saha_ratio / (1 + saha_ratio)
            neutral_fraction = 1 / (1 + saha_ratio)

            for ion_state in [1, 2]:
                nist_data, _ = fetcher.get_nist_data(elem, ion_state)
                if not nist_data: continue
                temp_sim = SpectrumSimulator(nist_data, elem, ion_state, temp, 0)
                levels, degeneracies = {}, {}
                for line in nist_data:
                    try:
                        Ei, gi, Ek, gk = float(line[3]), float(line[4]), float(line[2]), float(line[5])
                        levels[Ei], degeneracies[Ei] = Ei, gi
                        levels[Ek], degeneracies[Ek] = Ek, gk
                    except (ValueError, TypeError): continue
                Z = temp_sim.partition_function(list(levels.values()), list(degeneracies.values()))
                saha_factor = neutral_fraction if ion_state == 1 else ion_fraction
                
                calculated_lines_for_state = []
                for line in nist_data:
                    try:
                        line_data_dict = {
                            'λ (nm)': float(line[0]), 'E_k (eV)': float(line[2]),
                            'g_k': int(line[5]), 'A_ki(s^-1)': float(line[1])
                        }
                        i_boltzmann = temp_sim.calculate_intensity(line_data_dict['E_k (eV)'], line_data_dict['g_k'], line_data_dict['A_ki(s^-1)'], Z)
                        line_data_dict['Intensitas Final'] = i_boltzmann * saha_factor
                        calculated_lines_for_state.append(line_data_dict)
                    except (ValueError, TypeError): continue
                
                if calculated_lines_for_state:
                    df_state = pd.DataFrame(calculated_lines_for_state)
                    top_lines = df_state.nlargest(num_lines_per_state, 'Intensitas Final')
                    for _, row in top_lines.iterrows():
                        final_results.append({
                            'Elemen': elem,
                            'Garis Spektral': f"{elem} {'I' if ion_state == 1 else 'II'} @ {row['λ (nm)']:.2f} nm",
                            'Suhu (K)': temp,
                            'Intensitas Final': row['Intensitas Final']
                        })

    if not final_results:
        print("Tidak ada hasil yang dapat ditampilkan.")
        return
        
    df_final = pd.DataFrame(final_results)
    pivot_table = df_final.pivot_table(
        index=['Elemen', 'Garis Spektral'],
        columns='Suhu (K)',
        values='Intensitas Final'
    ).fillna(0)

    pivot_table = pivot_table.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # --- PERUBAHAN UTAMA: Normalisasi per kolom suhu di dalam setiap grup elemen ---
    # Hitung nilai maksimum untuk setiap grup elemen di setiap kolom suhu
    max_per_temp_col = pivot_table.groupby('Elemen').transform('max')
    
    # Lakukan normalisasi
    normalized_pivot = pivot_table / max_per_temp_col
    # Ganti nilai NaN yang mungkin muncul jika max_per_temp_col adalah 0
    normalized_pivot = normalized_pivot.fillna(0)


    print("\n" + "="*90)
    print("Tabel Hasil Analisis Parametrik: Intensitas Relatif Ternormalisasi vs Suhu")
    print("="*90)
    print("Intensitas dinormalisasi ke nilai maksimum PADA SETIAP SUHU untuk setiap elemen.")
    print(normalized_pivot.to_string(float_format="%.3f"))


if __name__ == "__main__":
    ELEMENTS_TO_STUDY = ['Al', 'Ca', 'Fe', 'Si', 'Mg']
    TEMPERATURES_TO_STUDY = [6000, 8000, 10000, 12000]
    ELECTRON_DENSITY = 1e17

    nist_data_file = os.path.join(script_dir, "data", "nist_data(1).h5")
    atomic_data_file = os.path.join(script_dir, "data", "atomic_data1.h5")

    if setup_atomic_data(atomic_data_file):
        perform_temperature_study(
            ELEMENTS_TO_STUDY,
            TEMPERATURES_TO_STUDY,
            ELECTRON_DENSITY,
            nist_data_file,
            num_lines_per_state=2
        )