# >>> Kode diagnostik di awal sim2.py <<<
import sys
import os
import json
import re
import argparse # Pastikan import ini ada di bagian atas file Anda

print("--- DIAGNOSTIK AWAL SKRIP ---")
print(f"Versi Python yang digunakan skrip: {sys.version}")
print(f"Executable Python yang digunakan: {sys.executable}")
print(f"PYTHONPATH saat ini: {os.environ.get('PYTHONPATH', 'Tidak diatur')}")
print(f"Path pencarian modul sys.path:")
for p_idx, p_val in enumerate(sys.path):
    print(f"  [{p_idx}] - {p_val}")

try:
    print("Mencoba mengimpor IPEX di dalam skrip (awal)...")
    import intel_extension_for_pytorch as ipex_diag
    print(f"IPEX berhasil diimpor di skrip (awal)! Versi: {ipex_diag.__version__}")
except ImportError as e:
    print(f"IPEX GAGAL diimpor di dalam skrip (awal). Error: {e}")
except Exception as e:
    print(f"Error lain saat mencoba impor IPEX di skrip (awal): {e}")
print("--- AKHIR DIAGNOSTIK AWAL SKRIP ---")
print("\nMelanjutkan eksekusi skrip asli...\n")
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F
from scipy.signal.windows import gaussian
import h5py
from tqdm import tqdm
import shutil
from typing import List, Dict, Tuple, Optional
from collections import Counter
import hashlib
from datetime import datetime
import logging
from multiprocessing import Pool
import numpy as np
import torch 
import torch.nn.functional as F
from scipy.signal.windows import gaussian
import json
import h5py
import os
import sys
import logging
import argparse
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

# Impor IPEX dengan logging versi
try:
    import intel_extension_for_pytorch as ipex
    ipex_version = getattr(ipex, '__version__', 'unknown')
except ImportError:
    ipex = None
    ipex_version = 'not installed'

# Konfigurasi logging
def setup_logging(base_dir: str, job_id: str = "unknown"):
    """Konfigurasi logger untuk stdout dan file dengan format terstruktur."""
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"spectral_simulation_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger('SpectralSimulation')
    logger.setLevel(logging.DEBUG)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(processName)s] %(message)s')
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.handlers.clear()
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    
    return logger

# Konfigurasi simulasi
SIMULATION_CONFIG = {
    "resolution": 4096,
    "wl_range": (200, 900),
    "sigma": 0.1,
    "target_max_intensity": 0.8,
    "convolution_sigma": 0.1,
    "num_samples": 2500,
    "temperature_range": np.linspace(5000, 25000, 100).tolist(),
    "electron_density_range": np.logspace(14, 18, 100).tolist(),
    "data_dir": "/home/bwalidain/birrulwldain/data",
    "processed_dir": "/home/bwalidain/_scratch/",
    "logs_dir": "/home/bwalidain/birrulwldain/logs"
}

# Konstanta fisika
PHYSICAL_CONSTANTS = {
    "k_B": 8.617333262145e-5,  # eV/K
    "m_e": 9.1093837e-31,      # kg
    "h": 4.135667696e-15,      # eV·s
}

# Elemen dan ion
BASE_ELEMENTS = ["Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl", "Mg", "C", "S", "Ar", "Ti", "Mn", "Co"]
REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]
class DataFetcher:
    """Mengambil data spektral dari file HDF5 NIST untuk elemen dan ion tertentu."""
    def __init__(self, hdf_path: str):
        self.hdf_path = hdf_path
        self.delta_E_max: Dict[str, float] = {}
        self.missing_data_count: Dict[str, int] = {}
        self.logger = logging.getLogger('SpectralSimulation')

    def get_nist_data(self, element: str, sp_num: int) -> Tuple[List[List], float]:
        elem_key = f"{element}_{sp_num}"
        try:
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                df = store.get('nist_spectroscopy_data')
                filtered_df = df[(df['element'] == element) & (df['sp_num'] == sp_num)]
                required_columns = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']

                if filtered_df.empty or not all(col in df.columns for col in required_columns):
                    self.missing_data_count[elem_key] = self.missing_data_count.get(elem_key, 0) + 1
                    if self.missing_data_count[elem_key] <= 3:
                        self.logger.warning(f"Tidak ada data untuk {elem_key} di dataset NIST")
                    return [], 0.0

                filtered_df = filtered_df.dropna(subset=required_columns)

                filtered_df['ritz_wl_air(nm)'] = pd.to_numeric(filtered_df['ritz_wl_air(nm)'], errors='coerce')
                for col in ['Ek(eV)', 'Ei(eV)', 'Aki(s^-1)', 'g_i', 'g_k']:
                    filtered_df[col] = pd.to_numeric(
                        filtered_df[col].apply(lambda x: re.sub(r'[^\d.-]', '', str(x)) if re.sub(r'[^\d.-]', '', str(x)) else None),
                        errors='coerce'
                    )

                filtered_df = filtered_df.dropna(subset=['ritz_wl_air(nm)', 'Ek(eV)', 'Ei(eV)', 'Aki(s^-1)', 'g_i', 'g_k'])

                filtered_df = filtered_df[
                    (filtered_df['ritz_wl_air(nm)'] >= SIMULATION_CONFIG["wl_range"][0]) &
                    (filtered_df['ritz_wl_air(nm)'] <= SIMULATION_CONFIG["wl_range"][1])
                ]

                filtered_df['delta_E'] = abs(filtered_df['Ek(eV)'] - filtered_df['Ei(eV)'])
                if filtered_df.empty:
                    self.missing_data_count[elem_key] = self.missing_data_count.get(elem_key, 0) + 1
                    if self.missing_data_count[elem_key] <= 3:
                        self.logger.warning(f"Tidak ada transisi valid untuk {elem_key} di rentang panjang gelombang")
                    return [], 0.0

                filtered_df = filtered_df.sort_values(by='Aki(s^-1)', ascending=False)
                delta_E_max = filtered_df['delta_E'].max()
                delta_E_max = 0.0 if pd.isna(delta_E_max) else delta_E_max
                self.delta_E_max[elem_key] = delta_E_max

                return filtered_df[required_columns + ['Acc']].values.tolist(), delta_E_max
        except Exception as e:
            self.logger.error(f"Error mengambil data NIST untuk {elem_key}: {str(e)}")
            return [], 0.0

class SpectrumSimulator:
    """
    Mensimulasikan spektrum emisi untuk satu elemen dan ion.
    REVISI: Menghapus metode .simulate() yang berlebihan dan memperbaiki normalisasi Gaussian.
    """
    def __init__(
        self,
        nist_data: List[List],
        element: str,
        ion: int,
        ionization_energy: float,
        config: Dict,
        element_map_labels: Dict[str, List[float]]
    ):
        self.nist_data = nist_data
        self.element = element
        self.ion = ion
        self.ionization_energy = ionization_energy
        self.resolution = config["resolution"]
        self.wl_range = config["wl_range"]
        self.sigma = config["sigma"]
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.gaussian_cache: Dict[float, np.ndarray] = {}
        self.element_label = f"{element}_{ion}"
        self.device = torch.device("cpu")
        self.element_map_labels = element_map_labels
        self.logger = logging.getLogger('SpectralSimulation')
        
        # Opsi untuk menggunakan tensor jika IPEX terinstal
        if 'ipex' in sys.modules:
            self.wavelengths = torch.tensor(self.wavelengths, device=self.device, dtype=torch.float32)

    def _partition_function(self, energy_levels: List[float], degeneracies: List[float], temperature: float) -> float:
        k_B = 8.617333262145e-5 # PHYSICAL_CONSTANTS["k_B"]
        return sum(g * np.exp(-E / (k_B * temperature)) for g, E in zip(degeneracies, energy_levels) if E is not None) or 1.0

    def _calculate_intensity(self, temperature: float, energy: float, degeneracy: float, einstein_coeff: float, Z: float) -> float:
        k_B = 8.617333262145e-5 # PHYSICAL_CONSTANTS["k_B"]
        return (degeneracy * einstein_coeff * np.exp(-energy / (k_B * temperature))) / Z

    def _gaussian_profile(self, center: float) -> np.ndarray:
        """
        --- REVISI: Normalisasi puncak ke 1 ---
        Profil dinormalisasi agar puncaknya bernilai 1. Intensitas sebenarnya
        ditentukan oleh _calculate_intensity.
        """
        if center not in self.gaussian_cache:
            # Gunakan torch jika self.wavelengths adalah tensor, jika tidak gunakan numpy
            if isinstance(self.wavelengths, torch.Tensor):
                x_tensor = self.wavelengths
                center_tensor = torch.tensor(center, device=self.device, dtype=torch.float32)
                sigma_tensor = torch.tensor(self.sigma, device=self.device, dtype=torch.float32)
                
                gaussian = torch.exp(-0.5 * ((x_tensor - center_tensor) / sigma_tensor) ** 2)
                self.gaussian_cache[center] = gaussian.cpu().numpy().astype(np.float32)
            else: # Fallback ke NumPy jika IPEX tidak digunakan
                gaussian = np.exp(-0.5 * ((self.wavelengths - center) / self.sigma) ** 2)
                self.gaussian_cache[center] = gaussian.astype(np.float32)

        return self.gaussian_cache[center]

    # Ganti fungsi ini di dalam kelas SpectrumSimulator di file job.py Anda

# Ganti fungsi ini di dalam kelas SpectrumSimulator di file job.py Anda

    def simulate_single_temp(self, temp: float, atom_percentage: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Mensimulasikan spektrum hanya untuk satu suhu spesifik secara efisien.
        REVISI: Menggunakan normalisasi Gaussian sebagai PDF (Probability Density Function).
        """
        if not self.nist_data:
            return None
        
        levels = {}
        for data in self.nist_data:
            try:
                _, _, Ek, Ei, gi, gk, _ = data
                if all(v is not None for v in [Ek, Ei, gi, gk]):
                    levels[float(Ei)] = float(gi)
                    levels[float(Ek)] = float(gk)
            except (ValueError, TypeError):
                continue

        if not levels:
            self.logger.warning(f"Tidak ada tingkat energi valid untuk {self.element_label} pada suhu {temp}K.")
            return None

        energy_levels = list(levels.keys())
        degeneracies = list(levels.values())

        Z = self._partition_function(energy_levels, degeneracies, temp)
        
        intensities = torch.zeros(self.resolution, device=self.device, dtype=torch.float32)
        element_contributions = torch.zeros(self.resolution, device=self.device, dtype=torch.float32)
        
        wavelength_axis = self.wavelengths.cpu().numpy() if isinstance(self.wavelengths, torch.Tensor) else self.wavelengths
        wavelength_step = (self.wl_range[1] - self.wl_range[0]) / (self.resolution - 1)
        sigma_points = self.sigma / wavelength_step

        for data in self.nist_data:
            try:
                wl, Aki, Ek, _, _, gk, _ = data
                if all(v is not None for v in [wl, Aki, Ek, gk]):
                    wl = float(wl)
                    Aki = float(Aki)
                    Ek = float(Ek)
                    gk = float(gk)
                    
                    intensity = self._calculate_intensity(temp, Ek, gk, Aki, Z)
                    idx = np.searchsorted(wavelength_axis, wl)

                    if 0 <= idx < self.resolution:
                        # --- AWAL BLOK LOGIKA YANG DIPERBAIKI ---
                        kernel_size = int(6 * sigma_points) | 1
                        kernel_half_width = kernel_size // 2
                        kernel_x = np.arange(-kernel_half_width, kernel_half_width + 1)
                        
                        # --- PERUBAHAN DI SINI ---
                        # Menambahkan faktor normalisasi untuk PDF.
                        # Ini membuat total area di bawah kurva Gaussian menjadi 1.
                        norm_factor = 1.0 / (sigma_points * np.sqrt(2 * np.pi))
                        kernel = norm_factor * np.exp(-0.5 * (kernel_x / sigma_points)**2)
                        # --- AKHIR PERUBAHAN ---
                        
                        target_start = idx - kernel_half_width
                        target_end = idx + kernel_half_width + 1

                        valid_target_start = max(0, target_start)
                        valid_target_end = min(self.resolution, target_end)
                        valid_kernel_start = valid_target_start - target_start
                        valid_kernel_end = valid_kernel_start + (valid_target_end - valid_target_start)

                        if valid_target_start < valid_target_end:
                            final_kernel_slice = kernel[valid_kernel_start:valid_kernel_end]
                            contribution = torch.tensor(
                                intensity * atom_percentage * final_kernel_slice,
                                device=self.device,
                                dtype=torch.float32
                            )
                            intensities[valid_target_start:valid_target_end] += contribution
                            element_contributions[valid_target_start:valid_target_end] += contribution
                        
            except (ValueError, TypeError):
                continue
        
        return intensities.cpu().numpy(), element_contributions.cpu().numpy()

class MixedSpectrumSimulator:
    """
    Menggabungkan spektrum dari beberapa simulator.
    REVISI: Menggunakan pembuatan label yang divektorisasi.
    """
    def __init__(self, simulators: List[SpectrumSimulator], config: Dict, delta_E_max: Dict[str, float], element_map_labels: Dict[str, List[float]]):
        self.simulators = simulators
        self.config = config
        self.resolution = config["resolution"]
        self.wl_range = config["wl_range"]
        self.convolution_sigma = config["convolution_sigma"]
        self.delta_E_max = delta_E_max
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.device = torch.device("cpu")
        self.intensity_threshold = 0.0005 # Pastikan threshold sama dengan di planner.py
        self.current_T: float = 0.0
        self.current_n_e: float = 0.0
        self.element_map_labels = element_map_labels
        self.num_labels = len(next(iter(element_map_labels.values())))
        self.logger = logging.getLogger('SpectralSimulation')

    def _normalize_intensity(self, intensity: np.ndarray, target_max: float) -> np.ndarray:
        max_val = np.max(intensity)
        if max_val == 0:
            return intensity
        return (intensity / max_val * target_max).astype(np.float32)

    def _convolve_spectrum(self, spectrum: np.ndarray, sigma_nm: float) -> np.ndarray:
        # Menggunakan NumPy untuk konvolusi karena lebih sederhana dan tidak perlu GPU
        wavelength_step = (self.wavelengths[-1] - self.wavelengths[0]) / (len(self.wavelengths) - 1)
        sigma_points = sigma_nm / wavelength_step
        kernel_size = int(6 * sigma_points) | 1
        kernel = gaussian(kernel_size, sigma_points)
        kernel /= np.sum(kernel)
        return np.convolve(spectrum, kernel, mode='same').astype(np.float32)

    def _saha_ratio(self, ion_energy: float, temp: float, electron_density: float) -> float:
        # (Asumsikan konstanta fisika sudah didefinisikan)
        k_B_eV_K = 8.617333262e-5
        m_e_kg = 9.1093837e-31
        h_eV_s = 4.135667696e-15
        
        kT_eV = k_B_eV_K * temp
        saha_factor = 2 * (2 * np.pi * m_e_kg * kT_eV * 1.60218e-19 / (h_eV_s * 1.60218e-19)**2)**1.5
        saha_factor /= (electron_density * 1e6) # konversi n_e dari cm^-3 ke m^-3
        
        # Menggunakan np.exp untuk stabilitas numerik
        return saha_factor * np.exp(-ion_energy / kT_eV)

    def generate_sample(self, ionization_energies: Dict[str, float], selected_base_elements: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
        self.logger.debug(f"Menjalankan resep: T={self.current_T} K, n_e={self.current_n_e:.2e}, elemen={selected_base_elements}")
        temp = self.current_T
        electron_density = self.current_n_e

        # ... (Kode untuk menghitung atom_percentages_dict seperti di skrip asli Anda) ...
        # (Untuk singkatnya, bagian ini diasumsikan sama persis)
        # Mari kita asumsikan `atom_percentages_dict` dan `selected_simulators` telah dibuat.
        # Contoh dummy untuk alur logika:
        atom_percentages_dict = {}
        total_target_percentage = 0.0
        for base_elem in selected_base_elements:
            elem_neutral = f"{base_elem}_1"
            elem_ion = f"{base_elem}_2"
            ion_energy = ionization_energies.get(f"{base_elem} I", 7.0) # Default 7 eV
            saha_ratio = self._saha_ratio(ion_energy, temp, electron_density)
            total_percentage = np.random.uniform(0.1, 5.0) # Contoh persentase acak
            atom_percentages_dict[elem_neutral] = total_percentage * (1 / (1 + saha_ratio))
            atom_percentages_dict[elem_ion] = total_percentage * (saha_ratio / (1 + saha_ratio))
            total_target_percentage += total_percentage
        
        if total_target_percentage > 0:
            scaling_factor = 100.0 / total_target_percentage
            for key in atom_percentages_dict:
                atom_percentages_dict[key] *= scaling_factor
        
        selected_simulators = [sim for sim in self.simulators if sim.element_label in atom_percentages_dict]
        # --- Akhir bagian dummy ---

        if not selected_simulators:
            self.logger.warning(f"Tidak ada simulator valid untuk elemen yang dipilih.")
            return None

        mixed_spectrum = np.zeros(self.resolution, dtype=np.float32)
        element_contributions = np.zeros((len(selected_simulators), self.resolution), dtype=np.float32)

        for sim_idx, simulator in enumerate(selected_simulators):
            atom_percentage = atom_percentages_dict.get(simulator.element_label, 0.0)
            if atom_percentage > 0:
                result = simulator.simulate_single_temp(temp, atom_percentage)
                if result is not None:
                    spectrum, contrib = result
                    mixed_spectrum += spectrum
                    element_contributions[sim_idx] = contrib

        if np.max(mixed_spectrum) == 0:
            self.logger.warning(f"Tidak ada spektrum dihasilkan untuk resep ini (T={temp}K).")
            return None

        convolved_spectrum = self._convolve_spectrum(mixed_spectrum, self.config["convolution_sigma"])
        normalized_spectrum = self._normalize_intensity(convolved_spectrum, self.config["target_max_intensity"])

        # --- REVISI KRITIS: Pembuatan label yang divektorisasi untuk performa ---
        
        # 1. Buat matriks one-hot untuk semua simulator yang aktif dalam batch ini
        active_labels_matrix = np.array(
            [self.element_map_labels.get(sim.element_label) for sim in selected_simulators], 
            dtype=np.float32
        ) # Shape: (num_simulators, num_labels)
        
        # 2. Buat boolean mask di mana kontribusi melebihi threshold
        active_mask = element_contributions >= self.intensity_threshold # Shape: (num_simulators, resolution)
        
        # 3. Hitung label multi-hot dengan perkalian matriks (transpose mask dulu)
        labels = active_mask.T @ active_labels_matrix # Shape: (resolution, num_labels)
        
        # 4. Tentukan piksel mana yang memiliki setidaknya satu kontribusi aktif
        is_active = labels.sum(axis=1) > 0
        
        # 5. Ubah hasil penjumlahan menjadi biner (0 atau 1)
        labels = np.where(labels > 0, 1.0, 0.0).astype(np.float32)
        
        # 6. Atur label background di mana tidak ada elemen aktif
        background_label = np.array(self.element_map_labels["background"], dtype=np.float32)
        labels[~is_active] = background_label
        
        # --- Akhir Revisi ---

        final_atom_percentages = {k: float(v) for k, v in atom_percentages_dict.items()}
        final_atom_percentages['temperature'] = float(temp)
        final_atom_percentages['electron_density'] = float(electron_density)

        return self.wavelengths, normalized_spectrum, labels, final_atom_percentages

class WorkerDatasetGenerator:
    """Kelas ini mengelola eksekusi resep oleh proses-proses pekerja."""
    def __init__(self, config: Dict, element_map_labels: Dict[str, List[float]]):
        self.config = config
        self.element_map_labels = element_map_labels
        self.logger = logging.getLogger('SpectralSimulation')

    def _generate_task(self, args: Tuple) -> Optional[Tuple]:
        """Fungsi pembungkus untuk menjalankan satu resep oleh satu proses."""
        recipe, simulators, ionization_energies, delta_E_max_dict, element_map_labels = args
        
        try:
            # Setiap proses membuat instance simulator-nya sendiri
            mixed_simulator = MixedSpectrumSimulator(simulators, self.config, delta_E_max_dict, element_map_labels)
            mixed_simulator.current_T = recipe["temperature"]
            mixed_simulator.current_n_e = recipe["electron_density"]
            
            result = mixed_simulator.generate_sample(
                ionization_energies,
                recipe["base_elements"]
            )
            
            if result is None:
                self.logger.warning(f"Gagal menghasilkan spektrum untuk resep ID: {recipe['sample_id']}")
                return None
            
            wavelengths, spectrum, labels, atom_percentages = result
            # Sertakan metadata asli untuk disimpan
            atom_percentages['sample_id'] = recipe['sample_id']
            return spectrum, labels, atom_percentages
        except Exception as e:
            self.logger.error(f"Error pada task untuk resep ID {recipe.get('sample_id', 'N/A')}: {e}", exc_info=True)
            return None

    def process_recipes(self, recipes: List[Dict], simulators: List[SpectrumSimulator], ionization_energies: Dict[str, float], delta_E_max_dict: Dict[str, float], output_h5_path: str):
        """Memproses daftar resep dan menyimpan hasilnya ke file HDF5."""
        self.logger.info(f"Akan memproses {len(recipes)} resep dan menyimpan ke {output_h5_path}")

        spectra_list, labels_list, atom_percentages_list = [], [], []
        
        # Atur jumlah worker sesuai core yang tersedia atau batasan
        max_workers = min(16, os.cpu_count() or 1)
        self.logger.info(f"Menggunakan {max_workers} proses pekerja.")
        
        with Pool(processes=max_workers) as pool:
            tasks = [(recipe, simulators, ionization_energies, delta_E_max_dict, self.element_map_labels) for recipe in recipes]
            
            output_filename = os.path.basename(output_h5_path)
            for result in tqdm(pool.imap_unordered(self._generate_task, tasks), total=len(tasks), desc=f"Processing {output_filename}"):
                if result is not None:
                    spectrum, labels, atom_percentages = result
                    spectra_list.append(spectrum)
                    labels_list.append(labels)
                    atom_percentages_list.append(atom_percentages)
        
        if not spectra_list:
            self.logger.error("Tidak ada sampel valid yang dihasilkan dari resep yang diberikan.")
            return

        self.logger.info(f"Menyimpan {len(spectra_list)} spektrum yang dihasilkan...")
        with h5py.File(output_h5_path, 'w') as f:
            # Simpan wavelengths, cukup sekali
            wavelengths_data = np.linspace(self.config["wl_range"][0], self.config["wl_range"][1], self.config["resolution"], dtype=np.float32)
            f.create_dataset('wavelengths', data=wavelengths_data)
            
            # Simpan semua data ke dalam satu grup untuk digabung nanti
            data_grp = f.create_group('chunk_data')
            data_grp.create_dataset('spectra', data=np.array(spectra_list, dtype=np.float32), compression='gzip')
            data_grp.create_dataset('labels', data=np.array(labels_list, dtype=np.float32), compression='gzip')
            
            atom_percentages_encoded = [json.dumps(d).encode('utf-8') for d in atom_percentages_list]
            dt = h5py.special_dtype(vlen=str)
            data_grp.create_dataset('atom_percentages', data=np.array(atom_percentages_encoded), dtype=dt, compression='gzip')

        self.logger.info(f"Pekerjaan selesai. Data disimpan di {output_h5_path}")

class DataManager:
    """Mengelola pemuatan data, validasi, dan operasi file."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.data_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "output")
        self.nist_target_path = os.path.join(self.data_dir, "nist_data(1).h5")
        self.atomic_data_target_path = os.path.join(self.data_dir, "atomic_data1.h5")
        self.json_map_path = os.path.join(self.data_dir, "element-map-35.json")
        self.logger = logging.getLogger('SpectralSimulation')

    def load_element_map(self) -> Dict[str, List[float]]:
        if not os.path.exists(self.json_map_path):
            self.logger.error(f"element_map.json tidak ditemukan di {self.json_map_path}")
            raise FileNotFoundError(f"element_map.json tidak ditemukan")
        with open(self.json_map_path, 'r') as f:
            element_map = json.load(f)

        # Validasi bahwa setiap elemen memiliki vektor one-hot yang valid
        vector_length = None
        for elem, vector in element_map.items():
            if not isinstance(vector, list) or not all(isinstance(v, (int, float)) for v in vector):
                self.logger.error(f"Vektor one-hot tidak valid untuk {elem}: {vector}")
                raise ValueError(f"Vektor one-hot tidak valid untuk {elem}")
            if vector_length is None:
                vector_length = len(vector)
            elif len(vector) != vector_length:
                self.logger.error(f"Panjang vektor one-hot tidak konsisten untuk {elem}: {len(vector)} vs {vector_length}")
                raise ValueError(f"Panjang vektor one-hot tidak konsisten")
        self.logger.info(f"Element map dimuat dengan {len(element_map)} elemen, panjang vektor one-hot: {vector_length}")
        return element_map

    def load_ionization_energies(self) -> Dict[str, float]:
        ionization_energies = {}
        try:
            with h5py.File(self.atomic_data_target_path, 'r') as f:
                dset = f['elements']
                columns = dset.attrs.get('columns', ['At. num', 'Sp. Name', 'Ion Charge', 'El. Name',
                                                   'Prefix', 'Ionization Energy (eV)', 'Suffix'])
                data = [[item[0], item[1].decode('utf-8'), item[2].decode('utf-8'), item[3].decode('utf-8'),
                        item[4].decode('utf-8'), item[5], item[6].decode('utf-8')] for item in dset[:]]
                df_ionization = pd.DataFrame(data, columns=columns)

                species_col = ion_energy_col = None
                for col in columns:
                    if col.lower() in ['sp.', 'species', 'sp', 'element', 'sp. name']:
                        species_col = col
                    if 'ionization' in col.lower() and 'ev' in col.lower():
                        ion_energy_col = col

                if not species_col or not ion_energy_col:
                    self.logger.error(f"Kolom yang diperlukan tidak ditemukan. Tersedia: {list(df_ionization.columns)}")
                    raise KeyError(f"Kolom yang diperlukan tidak ditemukan")

                for _, row in df_ionization.iterrows():
                    try:
                        ionization_energies[row[species_col]] = float(row[ion_energy_col])
                    except (ValueError, TypeError):
                        self.logger.warning(f"Energi ionisasi tidak valid untuk {row[species_col]}, menggunakan 0.0 eV")
                        ionization_energies[row[species_col]] = 0.0

        except Exception as e:
            self.logger.error(f"Error memuat atomic_data1.h5: {str(e)}")
            raise

        for elem in REQUIRED_ELEMENTS:
            base_elem, ion = elem.split('_')
            ion_level = 'I' if ion == '1' else 'II'
            sp_name = f"{base_elem} {ion_level}"
            if sp_name not in ionization_energies:
                self.logger.warning(f"Tidak ada energi ionisasi untuk {sp_name}, menggunakan 0.0 eV")
                ionization_energies[sp_name] = 0.0

        return ionization_energies


def main(args):
    """Fungsi utama untuk menjalankan simulasi dalam mode pekerja."""
    
    job_id = os.path.basename(args.output_h5).replace('.h5', '')
    # (Asumsikan setup_logging, SIMULATION_CONFIG, dan kelas DataManager/DataFetcher ada)
    logger = setup_logging(base_dir=SIMULATION_CONFIG["logs_dir"], job_id=job_id)
    logger.info(f"Memulai PEKERJA simulasi spektral untuk file resep: {args.input_json}")

    # --- REVISI: Set thread ke 1 untuk efisiensi multiprocessing ---
    # Paralelisme datang dari jumlah PROSES (worker), bukan dari thread di dalam proses.
    # Ini mencegah perebutan sumber daya CPU dan membuat kinerja lebih efisien dan dapat diprediksi.
    torch.set_num_threads(1)

    logger.info("Memuat data pendukung (NIST, Ionization, Element Map)...")
    base_dir = SIMULATION_CONFIG["data_dir"]
    data_manager = DataManager(base_dir)
    element_map = data_manager.load_element_map()
    ionization_energies = data_manager.load_ionization_energies()
    
    fetcher = DataFetcher(data_manager.nist_target_path)
    nist_data_dict = {}
    delta_E_max_dict = {}
    
    # (Asumsikan REQUIRED_ELEMENTS sudah didefinisikan)
    for elem in tqdm(REQUIRED_ELEMENTS, desc="Fetching NIST Data"):
        element, ion = elem.split('_')
        data, delta_E = fetcher.get_nist_data(element, int(ion))
        nist_data_dict[elem] = data
        delta_E_max_dict[elem] = delta_E

    logger.info("Mempersiapkan simulator untuk setiap ion...")
    simulators = []
    for elem_key, nist_data in nist_data_dict.items():
        if nist_data:
            element, ion_str = elem_key.split('_')
            ion = int(ion_str)
            ion_name_suffix = 'I' if ion == 1 else 'II'
            ion_energy = ionization_energies.get(f"{element} {ion_name_suffix}", 0.0)
            simulator = SpectrumSimulator(
                nist_data=nist_data, 
                element=element, 
                ion=ion, 
                ionization_energy=ion_energy, 
                config=SIMULATION_CONFIG, 
                element_map_labels=element_map
            )
            simulators.append(simulator)
    logger.info(f"Total {len(simulators)} simulator berhasil dibuat.")

    logger.info(f"Membaca resep dari {args.input_json}...")
    try:
        with open(args.input_json, 'r') as f:
            recipes = json.load(f)
    except FileNotFoundError:
        logger.error(f"File resep tidak ditemukan: {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Gagal mem-parsing file JSON: {args.input_json}. Pastikan formatnya benar.")
        sys.exit(1)

    # (Asumsikan kelas WorkerDatasetGenerator ada dan benar)
    generator = WorkerDatasetGenerator(SIMULATION_CONFIG, element_map)
    generator.process_recipes(
        recipes=recipes, 
        simulators=simulators, 
        ionization_energies=ionization_energies, 
        delta_E_max_dict=delta_E_max_dict, 
        output_h5_path=args.output_h5
    )

    logger.info("Proses pekerja selesai.")

# (A
if __name__ == "__main__":
    # Setup untuk membaca argumen dari command line
    parser = argparse.ArgumentParser(description="Simulasi Spektral - Mode Pekerja")
    parser.add_argument('--input-json', type=str, required=True, 
                        help='Path ke file JSON berisi resep (jatah pekerjaan)')
    parser.add_argument('--output-h5', type=str, required=True, 
                        help='Path ke file HDF5 untuk menyimpan hasil')
    
    parsed_args = parser.parse_args()
    
    # Panggil fungsi main dengan argumen yang sudah di-parse
    main(parsed_args)
