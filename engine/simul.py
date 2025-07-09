import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal.windows import gaussian
import h5py
import re
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import argparse
import sys

# Simulation parameters configuration
SIMULATION_CONFIG = {
    "resolution": 24480,
    "wl_range": (200, 900),
    "sigma": 0.1,
    "target_max_intensity": 0.8,
    "convolution_sigma": 0.1,
}

# Physical constants
PHYSICAL_CONSTANTS = {
    "k_B": 8.617333262145e-5,  # eV/K
    "m_e": 9.1093837e-31,      # kg
    "h": 4.135667696e-15,      # eV·s
}

# Global variable for ionization energies
ionization_energies = {}

def calculate_lte_electron_density(temp: float, delta_E: float) -> float:
    """Calculates the minimum electron density for LTE."""
    return 1.6e12 * (temp ** 0.5) * (delta_E ** 3)

class DataFetcher:
    """Class to fetch and filter spectroscopic data from an HDF5 file."""
    def __init__(self, hdf_path: str):
        self.hdf_path = hdf_path
        self.delta_E_max = {}

    def get_nist_data(self, element: str, sp_num: int) -> Tuple[List[List], float]:
        """Fetches data for a specific element and ionization state."""
        try:
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                df = store.get('nist_spectroscopy_data')
                filtered_df = df[(df['element'] == element) & (df['sp_num'] == sp_num)].copy()
                required_columns = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']
                if filtered_df.empty or not all(col in df.columns for col in required_columns):
                    return [], 0.0
                
                filtered_df.dropna(subset=required_columns, inplace=True)
                filtered_df['ritz_wl_air(nm)'] = pd.to_numeric(filtered_df['ritz_wl_air(nm)'], errors='coerce')
                for col in ['Ek(eV)', 'Ei(eV)']:
                    filtered_df[col] = filtered_df[col].apply(lambda x: float(re.sub(r'[^\d.-]', '', str(x))) if isinstance(x, str) else x)
                
                filtered_df.dropna(subset=['ritz_wl_air(nm)', 'Ek(eV)', 'Ei(eV)'], inplace=True)
                filtered_df = filtered_df[
                    (filtered_df['ritz_wl_air(nm)'] >= SIMULATION_CONFIG["wl_range"][0]) &
                    (filtered_df['ritz_wl_air(nm)'] <= SIMULATION_CONFIG["wl_range"][1])
                ]
                
                if filtered_df.empty: return [], 0.0
                
                filtered_df['delta_E'] = abs(filtered_df['Ek(eV)'] - filtered_df['Ei(eV)'])
                delta_E_max_val = filtered_df['delta_E'].max()
                self.delta_E_max[f"{element}_{sp_num}"] = 0.0 if pd.isna(delta_E_max_val) else delta_E_max_val
                return filtered_df[required_columns + ['Acc']].values.tolist(), self.delta_E_max[f"{element}_{sp_num}"]
        except Exception as e:
            print(f"Error fetching NIST data for {element}_{sp_num}: {e}")
            return [], 0.0

class SpectrumSimulator:
    """Simulates the spectrum for a single ion type."""
    def __init__(self, nist_data: List[List], temperature: float, config: Dict = SIMULATION_CONFIG):
        self.nist_data = nist_data
        self.temperature = temperature
        self.config = config
        self.wavelengths = np.linspace(config["wl_range"][0], config["wl_range"][1], config["resolution"], dtype=np.float32)
        self.gaussian_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def partition_function(self, levels: Dict) -> float:
        """Calculates the partition function."""
        return sum(g * np.exp(-E / (PHYSICAL_CONSTANTS["k_B"] * self.temperature)) for E, g in levels.items()) or 1.0

    def calculate_intensity(self, Ek: float, gk: float, Aki: float, Z: float) -> float:
        """Calculates the intensity of a spectral line."""
        return (gk * Aki * np.exp(-Ek / (PHYSICAL_CONSTANTS["k_B"] * self.temperature))) / Z

    def gaussian_profile(self, center: float) -> np.ndarray:
        """Creates a Gaussian profile for line broadening."""
        if center not in self.gaussian_cache:
            x = torch.from_numpy(self.wavelengths).to(self.device)
            gauss = torch.exp(-0.5 * ((x - center) / self.config["sigma"])**2)
            self.gaussian_cache[center] = gauss.cpu().numpy()
        return self.gaussian_cache[center]

    def simulate(self, atom_percentage: float = 1.0) -> np.ndarray:
        """Runs the simulation to generate the spectrum."""
        spectrum = np.zeros(self.config["resolution"], dtype=np.float32)
        levels = {}
        for data in self.nist_data:
            try:
                _, _, Ek, Ei, gi, gk, _ = data
                levels.setdefault(float(Ei), float(gi)); levels.setdefault(float(Ek), float(gk))
            except (ValueError, TypeError): continue
        if not levels: return spectrum

        Z = self.partition_function(levels)
        for data in self.nist_data:
            try:
                wl, Aki, Ek, _, _, gk, _ = data
                intensity = self.calculate_intensity(float(Ek), float(gk), float(Aki), Z)
                spectrum += intensity * atom_percentage * self.gaussian_profile(float(wl))
            except (ValueError, TypeError): continue
        return spectrum

class MixedSpectrumSimulator:
    """Combines spectra from multiple simulators and applies plasma conditions."""
    def __init__(self, simulators: Dict[str, SpectrumSimulator], electron_density: float, delta_E_max: Dict[str, float], config: Dict = SIMULATION_CONFIG):
        self.simulators = simulators
        self.electron_density = electron_density
        self.delta_E_max = delta_E_max
        self.config = config
        self.wavelengths = np.linspace(config["wl_range"][0], config["wl_range"][1], config["resolution"], dtype=np.float32)

    def normalize(self, spectrum: np.ndarray) -> np.ndarray:
        """Normalizes the spectrum to the target intensity."""
        max_val = np.max(spectrum)
        return spectrum / max_val * self.config["target_max_intensity"] if max_val > 0 else spectrum

    def convolve_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Performs convolution for instrumental broadening."""
        wavelength_step = (self.wavelengths[-1] - self.wavelengths[0]) / (len(self.wavelengths) - 1)
        sigma_points = self.config["convolution_sigma"] / wavelength_step
        kernel_size = int(6 * sigma_points) | 1
        kernel = gaussian(kernel_size, sigma_points)
        return np.convolve(spectrum, kernel / np.sum(kernel), mode='same')

    def saha_ratio(self, ion_energy: float, temp: float, electron_density_m3: float) -> float:
        """Calculates the Saha ratio for ionization equilibrium."""
        k_B_j, h_j, m_e = 1.380649e-23, 6.62607015e-34, 9.1093837e-31
        ion_energy_j = ion_energy * 1.60218e-19
        return ((2*np.pi*m_e*k_B_j*temp)/(h_j**2))**1.5 * (2/electron_density_m3) * np.exp(-ion_energy_j/(k_B_j*temp))

    def generate_spectrum(self, selected_elements: List[Tuple[str, float]], temperature: float, deconstruct: bool = False):
        """Generates the final mixed spectrum."""
        delta_E_values = [v for v in self.delta_E_max.values() if v > 0]
        delta_E_max = max(delta_E_values) if delta_E_values else 4.0
        n_e_min = calculate_lte_electron_density(temperature, delta_E_max)
        adjusted_electron_density = max(self.electron_density, n_e_min)
        
        mixed_spectrum = np.zeros_like(self.wavelengths)
        deconstructed_spectra = {} if deconstruct else None
        atom_percentages = {}

        for base_elem, percentage in selected_elements:
            pure_element_spectrum = np.zeros_like(self.wavelengths)
            ion_energy = ionization_energies.get(f"{base_elem} I", 0)
            s_ratio = self.saha_ratio(ion_energy, temperature, adjusted_electron_density * 1e6) if ion_energy > 0 else 0
            fractions = {'1': 1/(1+s_ratio), '2': s_ratio/(1+s_ratio)}
            
            for ion_state in ['1', '2']:
                elem_label = f"{base_elem}_{ion_state}"
                atom_percentages[elem_label] = percentage * fractions[ion_state]
                if elem_label in self.simulators:
                    self.simulators[elem_label].temperature = temperature
                    ion_spectrum = self.simulators[elem_label].simulate(atom_percentages[elem_label] / 100.0)
                    pure_element_spectrum += ion_spectrum
            
            mixed_spectrum += pure_element_spectrum
            if deconstruct: deconstructed_spectra[base_elem] = pure_element_spectrum

        final_spectrum = self.normalize(self.convolve_spectrum(mixed_spectrum))
        final_params = {'temperature': temperature, 'electron_density': adjusted_electron_density, 'delta_E_max': delta_E_max, 'n_e_min': n_e_min, **atom_percentages}
        return self.wavelengths, final_spectrum, final_params, deconstructed_spectra

def plot_spectrum(wavelengths: np.ndarray, spectrum: np.ndarray, params: Dict, element_names: List[str]):
    """Creates and saves the main plot for the mixed spectrum."""
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
    upper_limit = np.max(spectrum) * 1.1 if np.max(spectrum) > 0 else 1.0
    ax.set_ylim(-0.05 * upper_limit, upper_limit)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
    ax.minorticks_on()
    
    metadata_keys = {'temperature', 'electron_density', 'delta_E_max', 'n_e_min'}
    comp_lines = [f"{key.replace('_', ' ')}: {val:.2f}%" for key, val in params.items() if key not in metadata_keys]
    
    info_text = (f"$T = {params['temperature']:.0f}$ K\n"
                 f"$n_e = {params['electron_density']:.1e}$ cm$^{{-3}}$\n"
                 "-----------------------------------\n"
                 r"$\bf{Composition}$" + " (Saha-adjusted):\n" + 
                 '\n'.join(comp_lines) + '\n\n' +
                 f"$\\Delta E_{{max}} = {params['delta_E_max']:.2f}$ eV\n" +
                 f"$n_{{e, min}} = {params['n_e_min']:.1e}$ cm$^{{-3}}$")
                 
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black'))

    plt.tight_layout()
    
    base_dir = "/Volumes/Private-C/Private-C-backup/skripsi-21"
    output_dir = os.path.join(base_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    elements_str = "-".join(sorted(element_names))
    temp_str = f"T{int(params['temperature'])}K"
    ne_str = f"ne{params['electron_density']:.0e}".replace('+', '')
    wl_str = f"wl{int(wavelengths[0])}-{int(wavelengths[-1])}nm"
    filename = f"{elements_str}_{temp_str}_{ne_str}_{wl_str}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✅ Main plot saved to: {save_path}")

def plot_deconstructed_spectrum(wavelengths: np.ndarray, final_spectrum: np.ndarray, deconstructed: Dict, params: Dict, element_names: List[str]):
    """Creates and saves the deconstructed plot per element with group scaling."""
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
    ax.plot(wavelengths, final_spectrum, color='black', linewidth=2.0, label='Final Mixed Spectrum')
    
    wavelength_step = (wavelengths[-1] - wavelengths[0]) / (len(wavelengths) - 1)
    sigma_points = SIMULATION_CONFIG["convolution_sigma"] / wavelength_step
    kernel_size = int(6 * sigma_points) | 1
    kernel = gaussian(kernel_size, sigma_points)
    kernel /= np.sum(kernel)

    convolved_pure_spectra = {elem: np.convolve(spec, kernel, mode='same') for elem, spec in deconstructed.items()}
    global_max_pure = max(np.max(spec) for spec in convolved_pure_spectra.values()) if convolved_pure_spectra else 0
    scaling_factor = SIMULATION_CONFIG["target_max_intensity"] / global_max_pure if global_max_pure > 0 else 1

    all_plotted_spectra = [final_spectrum]
    colors = plt.cm.viridis(np.linspace(0, 1, len(convolved_pure_spectra)))
    for i, (elem, spec) in enumerate(convolved_pure_spectra.items()):
        scaled_spec = spec * scaling_factor
        ax.plot(wavelengths, scaled_spec, color=colors[i], linestyle='--', linewidth=1.2, label=f'Contribution: {elem}')
        all_plotted_spectra.append(scaled_spec)
    
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Intensity (a.u.)')
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    
    overall_max = max(np.max(s) for s in all_plotted_spectra)
    upper_limit = overall_max * 1.1 if overall_max > 0 else 1.0
    ax.set_ylim(-0.05 * upper_limit, upper_limit)

    ax.grid(True, which='major', linestyle='--', alpha=0.5); ax.legend()
    ax.minorticks_on()

    plt.tight_layout()

    base_dir = "/Volumes/Private-C/Private-C-backup/skripsi-21"
    output_dir = os.path.join(base_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    elements_str = "-".join(sorted(element_names))
    temp_str = f"T{int(params['temperature'])}K"
    ne_str = f"ne{params['electron_density']:.0e}".replace('+', '')
    wl_str = f"wl{int(wavelengths[0])}-{int(wavelengths[-1])}nm"
    filename = f"{elements_str}_{temp_str}_{ne_str}_{wl_str}_dk.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✅ Deconstructed plot saved to: {save_path}")

def run_simulation(selected_elements: List[Tuple[str, float]], temperature: float, electron_density: float, nist_path: str, atomic_data_path: str, deconstruct: bool):
    """Main function to run the entire simulation workflow."""
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
    except FileNotFoundError:
        print(f"Error: Atomic data file not found at '{atomic_data_path}'"); sys.exit(1)
    except Exception as e:
        print(f"Error reading atomic data file: {e}"); sys.exit(1)

    fetcher = DataFetcher(nist_path)
    simulators = {}
    for elem, _ in selected_elements:
        for ion in [1, 2]:
            nist_data, _ = fetcher.get_nist_data(elem, ion)
            if nist_data:
                simulators[f"{elem}_{ion}"] = SpectrumSimulator(nist_data, temperature)
    
    if not simulators:
        print("Error: No valid data found. Cannot create spectrum."); return

    mixer = MixedSpectrumSimulator(simulators, electron_density, fetcher.delta_E_max, SIMULATION_CONFIG)
    wavelengths, final_spectrum, params, deconstructed_spectra = mixer.generate_spectrum(selected_elements, temperature, deconstruct)
    
    print("\n--- Simulation Parameters ---")
    print(f"Temperature: {params['temperature']:.0f} K, Final Electron Density: {params['electron_density']:.1e} cm^-3")
    
    element_names = [e[0] for e in selected_elements]
    plot_spectrum(wavelengths, final_spectrum, params, element_names)
    if deconstruct and deconstructed_spectra:
        plot_deconstructed_spectrum(wavelengths, final_spectrum, deconstructed_spectra, params, element_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a simulated LIBS spectrum for given elements and concentrations.")
    parser.add_argument("-e", "--element", action='append', help="Specify an element and its percentage, e.g., 'Fe:70'.", required=True)
    parser.add_argument("-T", "--temperature", type=float, default=10000.0, help="Plasma temperature in Kelvin.")
    parser.add_argument("-ne", "--electron_density", type=float, default=1e16, help="Electron density in cm^-3.")
    parser.add_argument("-w", "--wavelength_range", type=str, default=f"{SIMULATION_CONFIG['wl_range'][0]}:{SIMULATION_CONFIG['wl_range'][1]}", help="Specify the wavelength range in nm.")
    parser.add_argument("-dk", "--deconstruct", action="store_true", help="Plot the deconstruction of pure element spectra.")
    args = parser.parse_args()
    
    try:
        start_wl, end_wl = map(float, args.wavelength_range.split(':'))
        if start_wl >= end_wl: raise ValueError("Start wavelength must be less than end wavelength.")
        SIMULATION_CONFIG["wl_range"] = (start_wl, end_wl)
    except ValueError as e:
        print(f"Error: Invalid wavelength range format '{args.wavelength_range}'. {e}"); sys.exit(1)

    selected_elements = []
    total_percentage = 0.0
    for elem_str in args.element:
        try:
            parts = elem_str.split(':')
            if len(parts) != 2: raise ValueError("Format must be ELEMENT:PERCENTAGE")
            percentage = float(parts[1].strip())
            selected_elements.append((parts[0].strip(), percentage)); total_percentage += percentage
        except ValueError as e:
            print(f"Error parsing element argument '{elem_str}': {e}"); sys.exit(1)
    if abs(total_percentage - 100.0) > 1e-6:
        print(f"Error: Total percentage of elements must be 100%. Current sum: {total_percentage:.1f}%"); sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    nist_data_file = os.path.join(script_dir, "data", "nist_data(1).h5")
    atomic_data_file = os.path.join(script_dir, "data", "atomic_data1.h5")

    print(f"\nAttempting to run simulation with data from: {os.path.join(script_dir, 'data')}")
    print(f"Please ensure '{os.path.basename(nist_data_file)}' and '{os.path.basename(atomic_data_file)}' are in that folder.")

    try:
        run_simulation(selected_elements, args.temperature, args.electron_density, nist_data_file, atomic_data_file, args.deconstruct)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
