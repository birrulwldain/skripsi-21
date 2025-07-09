import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F
from scipy.signal.windows import gaussian
import h5py
import re
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
import argparse
import sys

# Konfigurasi parameter simulasi
SIMULATION_CONFIG = {
    "resolution": 4096,
    "wl_range": (200, 900), # Nilai default, dapat diubah oleh argumen baris perintah
    "sigma": 0.1,
    "target_max_intensity": 0.8,
    "convolution_sigma": 0.1,
}

# Konstanta fisika
PHYSICAL_CONSTANTS = {
    "k_B": 8.617333262145e-5,  # eV/K
    "m_e": 9.1093837e-31,      # kg
    "h": 4.135667696e-15,      # eV·s
}

# Global variable for ionization energies (populated in run_simulation)
ionization_energies = {}

def calculate_lte_electron_density(temp: float, delta_E: float) -> float:
    """
    Menghitung densitas elektron minimum untuk LTE berdasarkan suhu dan Delta E.
    """
    return 1.6e12 * (temp ** 0.5) * (delta_E ** 3)

class DataFetcher:
    def __init__(self, hdf_path: str):
        self.hdf_path = hdf_path
        self.delta_E_max = {}  # Menyimpan Delta E maksimum per elemen dan ion

    def get_nist_data(self, element: str, sp_num: int) -> Tuple[List[List], float]:
        """
        Mengambil data spektrum untuk elemen dan ion, dan menghitung Delta E maksimum.
        """
        try:
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                df = store.get('nist_spectroscopy_data')
                filtered_df = df[(df['element'] == element) & (df['sp_num'] == sp_num)]
                required_columns = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']
                if filtered_df.empty or not all(col in df.columns for col in required_columns):
                    # Ini bukan error, hanya informasi bahwa data tidak ada
                    return [], 0.0
                filtered_df = filtered_df.dropna(subset=required_columns)

                filtered_df['ritz_wl_air(nm)'] = pd.to_numeric(filtered_df['ritz_wl_air(nm)'], errors='coerce')
                for col in ['Ek(eV)', 'Ei(eV)']:
                    filtered_df[col] = filtered_df[col].apply(
                        lambda x: float(re.sub(r'[^\d.-]', '', str(x))) if re.sub(r'[^\d.-]', '', str(x)) else None
                    )
                filtered_df = filtered_df.dropna(subset=['ritz_wl_air(nm)', 'Ek(eV)', 'Ei(eV)'])

                filtered_df = filtered_df[
                    (filtered_df['ritz_wl_air(nm)'] >= SIMULATION_CONFIG["wl_range"][0]) &
                    (filtered_df['ritz_wl_air(nm)'] <= SIMULATION_CONFIG["wl_range"][1])
                ]
                if filtered_df.empty:
                    return [], 0.0
                
                filtered_df['delta_E'] = abs(filtered_df['Ek(eV)'] - filtered_df['Ei(eV)'])
                filtered_df = filtered_df.sort_values(by='Aki(s^-1)', ascending=False)
                delta_E_max = filtered_df['delta_E'].max()
                if pd.isna(delta_E_max):
                    delta_E_max = 0.0
                
                self.delta_E_max[f"{element}_{sp_num}"] = delta_E_max
                return filtered_df[required_columns + ['Acc']].values.tolist(), delta_E_max
        except Exception as e:
            print(f"Error saat mengambil data NIST untuk {element}_{sp_num}: {str(e)}")
            return [], 0.0

class SpectrumSimulator:
    def __init__(
        self,
        nist_data: List[List],
        element: str,
        ion: int,
        temperature: float,
        ionization_energy: float,
        config: Dict = SIMULATION_CONFIG
    ):
        self.nist_data = nist_data
        self.element = element
        self.ion = ion
        self.temperature = temperature
        self.ionization_energy = ionization_energy
        self.resolution = config["resolution"]
        self.wl_range = config["wl_range"]
        self.sigma = config["sigma"]
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.gaussian_cache = {}
        self.element_label = f"{element}_{ion}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def partition_function(self, energy_levels: List[float], degeneracies: List[float]) -> float:
        k_B = PHYSICAL_CONSTANTS["k_B"]
        return sum(g * np.exp(-E / (k_B * self.temperature)) for g, E in zip(degeneracies, energy_levels) if E is not None) or 1.0

    def calculate_intensity(self, energy: float, degeneracy: float, einstein_coeff: float, Z: float) -> float:
        k_B = PHYSICAL_CONSTANTS["k_B"]
        return (degeneracy * einstein_coeff * np.exp(-energy / (k_B * self.temperature))) / Z

    def gaussian_profile(self, center: float) -> np.ndarray:
        if center not in self.gaussian_cache:
            x_tensor = torch.tensor(self.wavelengths, device=self.device, dtype=torch.float32)
            center_tensor = torch.tensor(center, device=self.device, dtype=torch.float32)
            sigma_tensor = torch.tensor(self.sigma, device=self.device, dtype=torch.float32)
            gaussian = torch.exp(-0.5 * ((x_tensor - center_tensor) / sigma_tensor) ** 2) / (sigma_tensor * torch.sqrt(torch.tensor(2 * np.pi, device=self.device)))
            self.gaussian_cache[center] = gaussian.cpu().numpy().astype(np.float32)
        return self.gaussian_cache[center]

    def simulate(self, atom_percentage: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        spectrum = torch.zeros(self.resolution, device=self.device, dtype=torch.float32)
        levels = {}

        for data in self.nist_data:
            try:
                wl, Aki, Ek, Ei, gi, gk, _ = data
                if all(v is not None for v in [wl, Aki, Ek, Ei, gi, gk]):
                    levels.setdefault(float(Ei), float(gi))
                    levels.setdefault(float(Ek), float(gk))
            except (ValueError, TypeError):
                continue

        if not levels:
            return self.wavelengths, np.zeros(self.resolution, dtype=np.float32)

        Z = self.partition_function(list(levels.keys()), list(levels.values()))

        for data in self.nist_data:
            try:
                wl, Aki, Ek, Ei, gi, gk, _ = data
                if all(v is not None for v in [wl, Aki, Ek, Ei, gi, gk]):
                    intensity = self.calculate_intensity(float(Ek), float(gk), float(Aki), Z)
                    gaussian_contrib = torch.tensor(
                        intensity * atom_percentage * self.gaussian_profile(float(wl)),
                        device=self.device,
                        dtype=torch.float32
                    )
                    spectrum += gaussian_contrib
            except (ValueError, TypeError):
                continue
        return self.wavelengths, spectrum.cpu().numpy()

class MixedSpectrumSimulator:
    def __init__(
        self,
        simulators: List[SpectrumSimulator],
        electron_density: float,
        delta_E_max: Dict[str, float],
        config: Dict = SIMULATION_CONFIG
    ):
        self.simulators = simulators
        self.resolution = config["resolution"]
        self.wl_range = config["wl_range"]
        self.convolution_sigma = config["convolution_sigma"]
        self.electron_density = electron_density
        self.delta_E_max = delta_E_max
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def normalize_intensity(self, intensity: np.ndarray, target_max: float) -> np.ndarray:
        intensity_tensor = torch.tensor(intensity, device=self.device, dtype=torch.float32)
        max_intensity = torch.max(torch.abs(intensity_tensor))
        return intensity if max_intensity == 0 else (intensity_tensor / max_intensity * target_max).cpu().numpy()

    def convolve_spectrum(self, spectrum: np.ndarray, sigma_nm: float) -> np.ndarray:
        spectrum_tensor = torch.from_numpy(spectrum).to(self.device).float().view(1, 1, -1)
        wavelength_step = (self.wavelengths[-1] - self.wavelengths[0]) / (len(self.wavelengths) - 1)
        sigma_points = sigma_nm / wavelength_step
        kernel_size = int(6 * sigma_points) | 1
        
        kernel_np = gaussian(kernel_size, sigma_points)
        kernel_np /= np.sum(kernel_np)
        kernel = torch.from_numpy(kernel_np).to(self.device).float().view(1, 1, -1)

        padding = kernel_size // 2
        convolved = F.conv1d(spectrum_tensor, kernel, padding=padding).squeeze().cpu().numpy()
        return convolved.astype(np.float32)

    def saha_ratio(self, ion_energy: float, temp: float, electron_density_m3: float) -> float:
        k_B_joules = 1.380649e-23 # J/K
        h_joules = 6.62607015e-34 # J·s
        m_e = PHYSICAL_CONSTANTS["m_e"]
        ion_energy_joules = ion_energy * 1.60218e-19
        
        saha_factor = ((2 * np.pi * m_e * k_B_joules * temp) / (h_joules**2))**1.5
        saha_factor *= (2 / electron_density_m3) * np.exp(-ion_energy_joules / (k_B_joules * temp))
        return saha_factor

    def validate_lte(self, temperature: float, selected_elements: List[Tuple[str, float]]) -> Tuple[float, float]:
        delta_E_values = [
            self.delta_E_max.get(f"{base_elem}_{ion}", 0.0)
            for base_elem, _ in selected_elements
            for ion in [1, 2]
        ]
        delta_E_max = max(delta_E_values) if any(v > 0 for v in delta_E_values) else 4.0
        n_e_min = calculate_lte_electron_density(temperature, delta_E_max)
        return delta_E_max, n_e_min

    def generate_spectrum(
        self,
        selected_elements: List[Tuple[str, float]],
        temperature: float
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        mixed_spectrum = np.zeros(self.resolution, dtype=np.float32)
        atom_percentages_dict = {}

        delta_E_max, n_e_min = self.validate_lte(temperature, selected_elements)
        adjusted_electron_density = max(self.electron_density, n_e_min)
        if self.electron_density < n_e_min:
            print(f"Peringatan: Densitas elektron {self.electron_density:.1e} cm^-3 di bawah syarat LTE.")
            print(f"Menggunakan n_e = {adjusted_electron_density:.1e} cm^-3 untuk memenuhi syarat LTE.")
        
        electron_density_m3 = adjusted_electron_density * 1e6

        for base_elem, percentage in selected_elements:
            ion_energy = ionization_energies.get(f"{base_elem} I", 0.0)
            if ion_energy == 0.0:
                print(f"Peringatan: Energi ionisasi untuk {base_elem} I tidak ditemukan.")
                continue
            
            s_ratio = self.saha_ratio(ion_energy, temperature, electron_density_m3)
            ion_fraction = s_ratio / (1 + s_ratio)
            neutral_fraction = 1 / (1 + s_ratio)
            
            atom_percentages_dict[f"{base_elem}_1"] = (percentage * neutral_fraction) / 100.0
            atom_percentages_dict[f"{base_elem}_2"] = (percentage * ion_fraction) / 100.0

        for simulator in self.simulators:
            if (elem_label := f"{simulator.element}_{simulator.ion}") in atom_percentages_dict:
                simulator.temperature = temperature
                _, spectrum = simulator.simulate(atom_percentages_dict[elem_label])
                mixed_spectrum += spectrum

        convolved_spectrum = self.convolve_spectrum(mixed_spectrum, self.convolution_sigma)
        normalized_spectrum = self.normalize_intensity(convolved_spectrum, SIMULATION_CONFIG["target_max_intensity"])
        
        final_percentages = {k: v * 100 for k, v in atom_percentages_dict.items()}
        final_percentages['temperature'] = float(temperature)
        final_percentages['electron_density'] = float(adjusted_electron_density)
        final_percentages['delta_E_max'] = float(delta_E_max)
        final_percentages['n_e_min'] = float(n_e_min)
        
        return self.wavelengths, normalized_spectrum, final_percentages

def plot_spectrum(wavelengths: np.ndarray, spectrum: np.ndarray, temperature: float, electron_density: float, atom_percentages: Dict):
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'lines.linewidth': 1.5, 'axes.linewidth': 1.2, 'xtick.major.size': 5,
        'ytick.major.size': 5, 'figure.dpi': 300, 'savefig.dpi': 300,
        'axes.facecolor': 'white', 'figure.facecolor': 'white',
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelengths, spectrum, color='navy', linewidth=1.5)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=16, labelpad=10)
    ax.set_ylabel('Normalized Intensity (a.u.)', fontsize=16, labelpad=10)
    
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    upper_limit = max(spectrum) * 1.1 if max(spectrum) > 0 else 1.0
    y_offset = -0.05 * upper_limit 
    ax.set_ylim(y_offset, upper_limit)
    
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
    ax.minorticks_on()
    
    base_elements = sorted(list(set([key.split('_')[0] for key in atom_percentages if '_' in key])))
    formatted_composition_lines = []
    for elem, perc in atom_percentages.items():
        if elem in ['temperature', 'electron_density', 'delta_E_max', 'n_e_min']:
            continue
        parts = elem.split('_')
        formatted_elem = f"{parts[0]} {'I' if parts[1] == '1' else 'II'}"
        formatted_composition_lines.append(f'{formatted_elem}: {perc:.2f}%')

    comp_text = (
        f'$T = {temperature:.0f}$ K\n'
        f'$n_e = {electron_density:.1e}$ cm$^{{-3}}$\n'
        '-----------------------------------\n'
        '**Komposisi (Saha-adjusted):**\n' +
        '\n'.join(formatted_composition_lines) + '\n\n' +
        f'Δ$E_{{max}}$: {atom_percentages["delta_E_max"]:.2f} eV\n' +
        f'$n_{{e,min}}$: {atom_percentages["n_e_min"]:.1e} cm$^{{-3}}$'
    )
    
    ax.text(0.98, 0.98, comp_text, 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black'))
    
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    base_dir = "/Volumes/Private-C/Private-C-backup/skripsi-21"
    output_dir = os.path.join(base_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    elements_str = "-".join(base_elements)
    temp_str = f"T{int(temperature)}K"
    ne_str = f"ne{electron_density:.0e}".replace('+', '')
    wl_str = f"wl{int(wavelengths[0])}-{int(wavelengths[-1])}nm"
    
    filename = f"{elements_str}_{temp_str}_{ne_str}_{wl_str}.png"
    output_file_path = os.path.join(output_dir, filename)

    plt.savefig(output_file_path, dpi=300, bbox_inches='tight', format='png')
    print(f"\n✅ Plot berhasil disimpan di: {output_file_path}")
    plt.close()

def run_simulation(selected_elements: List[Tuple[str, float]], temperature: float, electron_density: float,
                   nist_path: str, atomic_data_path: str):
    global ionization_energies
    ionization_energies = {}
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
    except FileNotFoundError:
        print(f"Error: File data atom tidak ditemukan di '{atomic_data_path}'")
        sys.exit(1)

    fetcher = DataFetcher(nist_path)
    simulators = []
    for elem, _ in selected_elements:
        for ion in [1, 2]:
            nist_data, _ = fetcher.get_nist_data(elem, ion)
            if nist_data:
                ion_energy = ionization_energies.get(f"{elem} {'I' if ion == 1 else 'II'}", 0.0)
                simulators.append(SpectrumSimulator(nist_data, elem, ion, temperature, ion_energy, SIMULATION_CONFIG))
            else:
                 print(f"Info: Tidak ada data NIST untuk {elem}_{ion} pada rentang panjang gelombang yang dipilih.")

    if not simulators:
        print("Error: Tidak ada data yang valid ditemukan. Spektrum tidak dapat dibuat.")
        return

    mixed_simulator = MixedSpectrumSimulator(simulators, electron_density, fetcher.delta_E_max, SIMULATION_CONFIG)
    wavelengths, spectrum, atom_percentages = mixed_simulator.generate_spectrum(selected_elements, temperature)

    print("\n--- Parameter Simulasi ---")
    print(f"Rentang Gelombang: {SIMULATION_CONFIG['wl_range'][0]} - {SIMULATION_CONFIG['wl_range'][1]} nm")
    print(f"Suhu: {temperature:.0f} K")
    print(f"Densitas Elektron Awal: {electron_density:.1e} cm^-3")
    print(f"Densitas Elektron Disesuaikan (untuk LTE): {atom_percentages['electron_density']:.1e} cm^-3")
    print(f"Celah Energi Maksimum (ΔE_max): {atom_percentages['delta_E_max']:.2f} eV")
    print(f"Densitas Elektron Minimum (n_e_min): {atom_percentages['n_e_min']:.1e} cm^-3")
    print("\n--- Komposisi Atom (setelah penyesuaian Saha) ---")
    for elem, percentage in atom_percentages.items():
        if elem not in ['temperature', 'electron_density', 'delta_E_max', 'n_e_min']:
            print(f"  {elem.replace('_1', ' I').replace('_2', ' II')}: {percentage:.2f}%")

    plot_spectrum(wavelengths, spectrum, temperature, atom_percentages['electron_density'], atom_percentages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buat simulasi spektrum LIBS untuk elemen dan konsentrasi tertentu.")
    parser.add_argument(
        "-e", "--element", action='append',
        help="Spesifikasi elemen dan persentasenya, misal 'Fe:70'. Ulangi untuk beberapa elemen.", required=True
    )
    parser.add_argument(
        "-T", "--temperature", type=float, default=10000.0,
        help="Suhu plasma dalam Kelvin (default: 10000.0)."
    )
    parser.add_argument(
        "-ne", "--electron_density", type=float, default=1e16,
        help="Densitas elektron dalam cm^-3 (default: 1e16)."
    )
    parser.add_argument(
        "-w", "--wavelength_range", type=str, default=f"{SIMULATION_CONFIG['wl_range'][0]}:{SIMULATION_CONFIG['wl_range'][1]}",
        help=f"Spesifikasi rentang panjang gelombang dalam nm, misal '370:400'. Default: '{SIMULATION_CONFIG['wl_range'][0]}:{SIMULATION_CONFIG['wl_range'][1]}'."
    )
    parser.add_argument(
        "-rc", "--random_concentrations", action="store_true",
        help="Buat konsentrasi acak untuk elemen yang ditentukan. Persentase pada argumen -e akan diabaikan."
    )
    args = parser.parse_args()
    
    try:
        start_wl, end_wl = map(float, args.wavelength_range.split(':'))
        if start_wl >= end_wl:
            raise ValueError("Panjang gelombang awal harus lebih kecil dari panjang gelombang akhir.")
        SIMULATION_CONFIG["wl_range"] = (start_wl, end_wl)
    except ValueError as e:
        print(f"Error: Format rentang panjang gelombang tidak valid '{args.wavelength_range}'. {e}")
        sys.exit(1)

    selected_elements = []
    if args.random_concentrations:
        base_elements = [elem_str.split(':')[0].strip() for elem_str in args.element]
        random_nums = np.random.rand(len(base_elements))
        normalized_percentages = (random_nums / np.sum(random_nums)) * 100.0
        selected_elements = list(zip(base_elements, normalized_percentages))
        print("\n--- Konsentrasi Acak Dihasilkan ---")
        for elem, perc in selected_elements:
            print(f"  {elem}: {perc:.2f}%")
    else:
        total_percentage = 0.0
        for elem_str in args.element:
            try:
                parts = elem_str.split(':')
                if len(parts) != 2: raise ValueError("Format harus ELEMEN:PERSENTASE")
                percentage = float(parts[1].strip())
                if not (0 <= percentage <= 100): raise ValueError("Persentase harus antara 0 dan 100.")
                selected_elements.append((parts[0].strip(), percentage))
                total_percentage += percentage
            except ValueError as e:
                print(f"Error saat mem-parsing argumen elemen '{elem_str}': {e}")
                sys.exit(1)
        if abs(total_percentage - 100.0) > 1e-6:
            print(f"Error: Total persentase elemen harus 100%. Jumlah saat ini: {total_percentage:.1f}%")
            sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    nist_data_file = os.path.join(data_dir, "nist_data(1).h5")
    atomic_data_file = os.path.join(data_dir, "atomic_data1.h5")

    print(f"\nMencoba menjalankan simulasi dengan data dari: {data_dir}")
    print(f"Pastikan '{os.path.basename(nist_data_file)}' dan '{os.path.basename(atomic_data_file)}' ada di folder tersebut.")

    try:
        run_simulation(selected_elements, args.temperature, args.electron_density,
                       nist_path=nist_data_file, atomic_data_path=atomic_data_file)
    except Exception as e:
        print(f"\nTerjadi error yang tidak terduga: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)