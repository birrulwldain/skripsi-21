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
# Fungsi-fungsi Tampilan (Display)
# ---------------------------------------------------------------------------

def display_saha_table(saha_results, temp, ne):
    """Menampilkan Tabel 1: Fraksi Ionisasi."""
    print("="*80)
    print(f"Tabel 1: Parameter Plasma & Fraksi Ionisasi (Saha) pada T={temp} K, ne={ne:.1e} cm⁻³")
    print("="*80)
    df = pd.DataFrame(saha_results)
    df['Fraksi Netral (%)'] = df['neutral_fraction'] * 100
    df['Fraksi Ion (%)'] = df['ion_fraction'] * 100
    print(df[['Elemen', 'Energi Ionisasi (eV)', 'Rasio Saha (N_II/N_I)', 'Fraksi Netral (%)', 'Fraksi Ion (%)']].to_string(index=False))
    print("\n")

def display_partition_and_lines_table(lines_to_display, partition_data):
    """Menampilkan Tabel 2: Fungsi Partisi dan detail garis transisi."""
    print("="*100)
    print("Tabel 2: Fungsi Partisi (Z) & Parameter 5 Garis Transisi Terkuat (dari NIST)")
    print("="*100)
    
    all_lines_formatted = []
    # Mengelompokkan berdasarkan elemen dan ion
    grouped = lines_to_display.groupby(['Elemen', 'Ion State'])
    for name, group in grouped:
        elem, ion_state = name
        ion_state_str = "I" if ion_state == 1 else "II"
        Z_value = partition_data.get(f"{elem}_{ion_state}", 0)

        # Tambahkan baris header untuk setiap grup
        all_lines_formatted.append({
            "Elemen": f"{elem} {ion_state_str}",
            "λ (nm)": f"Z = {Z_value:.2f}",
            "Ei (eV)": "-", "Ek (eV)": "-", "gi": "-", "gk": "-", "Aki (s⁻¹)": "-"
        })
        # Tambahkan baris data
        for _, row in group.iterrows():
            all_lines_formatted.append({
                "Elemen": "-",
                "λ (nm)": f"{row['obs_wl_air(nm)']:.3f}",
                "Ei (eV)": f"{row['E_i(eV)']:.3f}", "Ek (eV)": f"{row['E_k(eV)']:.3f}",
                "gi": int(row['g_i']), "gk": int(row['g_k']), "Aki (s⁻¹)": f"{row['A_ki(s^-1)']:.2e}"
            })

    df = pd.DataFrame(all_lines_formatted)
    print(df.to_string(index=False))
    print("\n")

def display_intensity_table(lines_to_display):
    """Menampilkan Tabel 3: Rincian perhitungan intensitas."""
    print("="*100)
    print("Tabel 3: Rincian Perhitungan Intensitas untuk 5 Garis Terkuat")
    print("="*100)
    
    # Format kolom untuk output yang rapi
    df_display = lines_to_display.copy()
    df_display['I_Boltzmann (rel)'] = df_display['I_Boltzmann (rel)'].map('{:.2e}'.format)
    df_display['Faktor Saha'] = df_display['Faktor Saha'].map('{:.3f}'.format)
    df_display['I_Final (rel)'] = df_display['I_Final (rel)'].map('{:.2e}'.format)
    df_display['I_Final (Normalized)'] = df_display['I_Final (Normalized)'].map('{:.3f}'.format)
    
    df_display['Elemen'] = df_display.apply(lambda row: f"{row['Elemen']} {'I' if row['Ion State'] == 1 else 'II'}", axis=1)
    df_display['λ (nm)'] = df_display['obs_wl_air(nm)'].map('{:.3f}'.format)
    
    print(df_display[['Elemen', 'λ (nm)', 'I_Boltzmann (rel)', 'Faktor Saha', 'I_Final (rel)', 'I_Final (Normalized)']].to_string(index=False))
    print("\n")


if __name__ == "__main__":
    # --- KONFIGURASI UTAMA ---
    elements_to_show = ['Al', 'Ca']
    TEMPERATURE = 8000
    ELECTRON_DENSITY = 1e17
    NUM_LINES = 5

    # Setup path file
    nist_data_file = os.path.join(script_dir, "data", "nist_data(1).h5")
    atomic_data_file = os.path.join(script_dir, "data", "atomic_data1.h5")

    if not setup_atomic_data(atomic_data_file):
        sys.exit(1)

    fetcher = DataFetcher(nist_data_file)
    mixed_sim = MixedSpectrumSimulator([], ELECTRON_DENSITY, {})
    all_calculated_lines = []
    saha_results = []
    partition_data = {}

    print("Memulai perhitungan fisika untuk SEMUA garis transisi...")
    
    for elem in elements_to_show:
        ion_energy = ionization_energies.get(f"{elem} I")
        saha_ratio = mixed_sim.saha_ratio(ion_energy, TEMPERATURE, ELECTRON_DENSITY * 1e6)
        ion_fraction = saha_ratio / (1 + saha_ratio)
        neutral_fraction = 1 / (1 + saha_ratio)
        saha_results.append({
            "Elemen": elem, "Energi Ionisasi (eV)": f"{ion_energy:.3f}",
            "Rasio Saha (N_II/N_I)": f"{saha_ratio:.2e}",
            "neutral_fraction": neutral_fraction, "ion_fraction": ion_fraction
        })

    for elem in elements_to_show:
        for ion_state in [1, 2]:
            nist_data, _ = fetcher.get_nist_data(elem, ion_state)
            if not nist_data: continue

            temp_sim = SpectrumSimulator(nist_data, elem, ion_state, TEMPERATURE, 0)
            
            levels, degeneracies = {}, {}
            for line in nist_data:
                try:
                    Ei, gi, Ek, gk = float(line[3]), float(line[4]), float(line[2]), float(line[5])
                    levels[Ei], degeneracies[Ei] = Ei, gi
                    levels[Ek], degeneracies[Ek] = Ek, gk
                except (ValueError, TypeError): continue
            
            Z = temp_sim.partition_function(list(levels.values()), list(degeneracies.values()))
            partition_data[f"{elem}_{ion_state}"] = Z
            saha_factor = next(item['neutral_fraction'] for item in saha_results if item["Elemen"] == elem) if ion_state == 1 else next(item['ion_fraction'] for item in saha_results if item["Elemen"] == elem)
            
            for line in nist_data:
                try:
                    line_data_dict = {
                        'Elemen': elem, 'Ion State': ion_state,
                        'obs_wl_air(nm)': float(line[0]), 'A_ki(s^-1)': float(line[1]),
                        'E_k(eV)': float(line[2]), 'E_i(eV)': float(line[3]),
                        'g_i': int(line[4]), 'g_k': int(line[5])
                    }
                    i_boltzmann = temp_sim.calculate_intensity(line_data_dict['E_k(eV)'], line_data_dict['g_k'], line_data_dict['A_ki(s^-1)'], Z)
                    line_data_dict['I_Boltzmann (rel)'] = i_boltzmann
                    line_data_dict['Faktor Saha'] = saha_factor
                    line_data_dict['I_Final (rel)'] = i_boltzmann * saha_factor
                    all_calculated_lines.append(line_data_dict)
                except (ValueError, TypeError): continue
    
    df_all_lines = pd.DataFrame(all_calculated_lines)
    max_intensity = df_all_lines['I_Final (rel)'].max()
    if max_intensity > 0:
        df_all_lines['I_Final (Normalized)'] = df_all_lines['I_Final (rel)'] / max_intensity
    else:
        df_all_lines['I_Final (Normalized)'] = 0.0

    # --- PERUBAHAN UTAMA: Filter garis untuk ditampilkan (berdasarkan I_Final tertinggi) ---
    lines_to_display = df_all_lines.groupby(['Elemen', 'Ion State'], group_keys=False).apply(
        lambda x: x.nlargest(NUM_LINES, 'I_Final (rel)')
    )

    # Tampilkan semua tabel
    print("\n--- HASIL PERHITUNGAN ---")
    display_saha_table(saha_results, TEMPERATURE, ELECTRON_DENSITY)
    display_partition_and_lines_table(lines_to_display, partition_data)
    display_intensity_table(lines_to_display)