# Nama file: planner.py (Versi Imbang untuk 10–17 Elemen per Sampel)

import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm
import logging
import argparse
import math
from collections import defaultdict

# --- Konfigurasi ---
NIST_HDF_PATH = "/home/bwalidain/birrulwldain/data/nist_data(1).h5"  # Sesuaikan path ini
BASE_ELEMENTS = ["Al", "Ar", "C", "Ca", "Cl", "Co", "Cr", "Fe", "Mg", "Mn", "N", "Na", "Ni", "O", "S", "Si", "Ti"]
TEMPERATURE_RANGE = np.linspace(5000, 15000, 100).tolist()
ELECTRON_DENSITY_RANGE = np.logspace(14, 18, 100).tolist()

# --- Konfigurasi untuk Estimasi Efisiensi Posisi ---
SIMULATION_RESOLUTION = 4096
SIMULATION_WL_RANGE = (200, 900)
SIMULATION_SIGMA = 0.1  # Sigma untuk profil Gaussian garis individual
SIMULATION_INTENSITY_THRESHOLD = 0.0005  # Pastikan sama dengan job.py

# --- Setup Logging Sederhana ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Fungsi calculate_sampling_weights ---
def calculate_sampling_weights():
    """
    Menghitung estimasi posisi label aktif per elemen berdasarkan data NIST.
    Mengembalikan estimasi posisi aktif per elemen untuk alokasi sampel.
    """
    logging.info("Membaca data NIST untuk estimasi efisiensi posisi label...")
    if not os.path.exists(NIST_HDF_PATH):
        raise FileNotFoundError(f"File data NIST tidak ditemukan di: {NIST_HDF_PATH}")
    
    with pd.HDFStore(NIST_HDF_PATH, mode='r') as store:
        df = store.get('nist_spectroscopy_data')
    
    # Siapkan sumbu panjang gelombang
    wavelengths = np.linspace(SIMULATION_WL_RANGE[0], SIMULATION_WL_RANGE[1], SIMULATION_RESOLUTION, dtype=np.float32)
    
    # Cache untuk profil Gaussian
    gaussian_cache = {}

    def _gaussian_profile(center_wl: float, wl_axis: np.ndarray, sigma: float) -> np.ndarray:
        wl_step = (wl_axis[-1] - wl_axis[0]) / (len(wl_axis) - 1)
        sigma_points = sigma / wl_step
        cache_key = (center_wl, sigma_points)
        if cache_key not in gaussian_cache:
            gaussian_profile_array = np.exp(-0.5 * ((wl_axis - center_wl) / sigma_points) ** 2)
            max_val = np.max(gaussian_profile_array)
            if max_val > 0:
                gaussian_profile_array /= max_val
            gaussian_cache[cache_key] = gaussian_profile_array
        return gaussian_cache[cache_key]

    logging.info("Mulai estimasi posisi label aktif per spesies ion dari data NIST...")
    
    estimated_active_positions_per_ion = defaultdict(float) 
    filtered_df = df[df['sp_num'].isin([1, 2])]

    for elem in tqdm(BASE_ELEMENTS, desc="Estimasi Efisiensi Elemen"):
        for sp_num in [1, 2]:
            ion_key = f"{elem}_{sp_num}"
            elem_ion_df = filtered_df[(filtered_df['element'] == elem) & (filtered_df['sp_num'] == sp_num)]
            
            ion_spectrum_temp = np.zeros(SIMULATION_RESOLUTION, dtype=np.float32)

            for index, row in elem_ion_df.iterrows():
                try:
                    wl = pd.to_numeric(row['ritz_wl_air(nm)'], errors='coerce')
                    aki = pd.to_numeric(row['Aki(s^-1)'], errors='coerce')
                    
                    if pd.notna(wl) and pd.notna(aki) and wl >= SIMULATION_WL_RANGE[0] and wl <= SIMULATION_WL_RANGE[1]:
                        peak_intensity_proxy = float(aki)
                        gaussian_profile = _gaussian_profile(wl, wavelengths, SIMULATION_SIGMA)
                        ion_spectrum_temp += peak_intensity_proxy * gaussian_profile
                except Exception as e:
                    logging.warning(f"Error memproses garis {ion_key} ({row['ritz_wl_air(nm)']}, {row['Aki(s^-1)']}): {e}")
                    continue

            active_positions_count = np.sum(ion_spectrum_temp > SIMULATION_INTENSITY_THRESHOLD)
            estimated_active_positions_per_ion[ion_key] = float(active_positions_count)
            
            if active_positions_count == 0 and not elem_ion_df.empty:
                logging.warning(f"Spesies ion '{ion_key}' memiliki garis namun tidak menghasilkan posisi aktif.")

    logging.info("Estimasi posisi label aktif per spesies ion selesai:")
    for ion_key, count in sorted(estimated_active_positions_per_ion.items(), key=lambda item: item[0]):
        logging.info(f"  - {ion_key}: {count:.0f} posisi aktif")
        
    # Estimasi total per elemen
    estimated_active_positions_per_element_total = {}
    for elem in BASE_ELEMENTS:
        ion1_key = f"{elem}_1"
        ion2_key = f"{elem}_2"
        total_active = estimated_active_positions_per_ion.get(ion1_key, 0.0) + \
                       estimated_active_positions_per_ion.get(ion2_key, 0.0)
        estimated_active_positions_per_element_total[elem] = max(total_active, 1.0)  # Default ke 1 jika 0

    logging.info("Estimasi posisi label aktif total per elemen dasar:")
    for elem, total_active in sorted(estimated_active_positions_per_element_total.items(), key=lambda item: item[0]):
        logging.info(f"  - {elem}: {total_active:.0f} posisi aktif total")

    return estimated_active_positions_per_element_total

# --- Fungsi generate_combinations_plan ---
def generate_combinations_plan(num_samples, output_path):
    """
    Membuat rencana kombinasi untuk num_samples sampel, dengan setiap sampel berisi
    10–17 elemen, dan distribusi posisi label imbang antar elemen.
    """
    logging.info(f"Memulai pembuatan rencana kombinasi untuk {num_samples} sampel...")
    
    # Estimasi posisi aktif per elemen
    active_positions = calculate_sampling_weights()
    
    # Target posisi per elemen (background ~20%)
    total_positions = num_samples * SIMULATION_RESOLUTION  # 10,000 * 4,096 = 40,960,000
    active_positions_total = total_positions * 0.8  # 80% untuk elemen = 32,768,000
    target_positions_per_element = active_positions_total / len(BASE_ELEMENTS)  # ~1,927,529
    
    # Hitung frekuensi kemunculan elemen di seluruh sampel
    appearances_per_element = {
        e: math.ceil(target_positions_per_element / pos) for e, pos in active_positions.items()
    }
    
    # Rata-rata elemen per sampel (antara 10–17, target ~13.5)
    avg_elements_per_sample = 13.5
    total_appearances_needed = int(num_samples * avg_elements_per_sample)  # 10,000 * 13.5 = 135,000
    
    # Normalisasi frekuensi kemunculan
    total_appearances_current = sum(appearances_per_element.values())
    scaling_factor = total_appearances_needed / total_appearances_current
    appearances_per_element = {
        e: max(1, math.floor(n * scaling_factor)) for e, n in appearances_per_element.items()
    }
    
    # Distribusikan sisa kemunculan
    current_total = sum(appearances_per_element.values())
    if current_total < total_appearances_needed:
        deficit = total_appearances_needed - current_total
        sorted_elements = sorted(appearances_per_element.items(), key=lambda x: active_positions[x[0]])
        for e, _ in sorted_elements[:deficit]:
            appearances_per_element[e] += 1
    elif current_total > total_appearances_needed:
        excess = current_total - total_appearances_needed
        sorted_elements = sorted(appearances_per_element.items(), key=lambda x: active_positions[x[0]], reverse=True)
        for e, _ in sorted_elements[:excess]:
            if appearances_per_element[e] > 1:
                appearances_per_element[e] -= 1
    
    logging.info("Frekuensi kemunculan elemen di seluruh sampel:")
    for e, n in sorted(appearances_per_element.items()):
        logging.info(f"  - {e}: {n} kemunculan (estimasi posisi aktif: {active_positions[e]:.0f})")
    
    # Generate resep dengan kombinasi elemen
    all_combinations = []
    sample_id = 0
    remaining_appearances = appearances_per_element.copy()
    elements_list = list(BASE_ELEMENTS)
    
    for _ in range(num_samples):
        # Tentukan jumlah elemen untuk sampel ini (10–17)
        num_elements = np.random.randint(10, 18)
        
        # Pilih elemen berdasarkan sisa kemunculan
        available_elements = [e for e in elements_list if remaining_appearances[e] > 0]
        if len(available_elements) < num_elements:
            # Reset kemunculan jika habis
            remaining_appearances = appearances_per_element.copy()
            available_elements = elements_list
        
        # Urutkan elemen berdasarkan sisa kemunculan (prioritas elemen dengan kemunculan tinggi)
        available_elements.sort(key=lambda e: remaining_appearances[e], reverse=True)
        selected_elements = available_elements[:num_elements]
        
        # Kurangi sisa kemunculan
        for e in selected_elements:
            remaining_appearances[e] -= 1
        
        # Generate resep
        temperature = np.random.choice(TEMPERATURE_RANGE)
        electron_density = np.random.choice(ELECTRON_DENSITY_RANGE)
        recipe = {
            "sample_id": sample_id,
            "temperature": float(temperature),
            "electron_density": float(electron_density),
            "base_elements": selected_elements
        }
        all_combinations.append(recipe)
        sample_id += 1
    
    logging.info(f"Total {len(all_combinations)} resep berhasil dibuat.")
    
    logging.info(f"Menyimpan rencana ke {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_combinations, f, indent=2)
        
    logging.info("Pembuatan rencana selesai.")
    return all_combinations

# --- Fungsi split_json_plan ---
def split_json_plan(all_recipes: list, base_filename: str, num_chunks: int):
    """
    Membagi daftar resep menjadi beberapa file chunk.
    """
    total_recipes = len(all_recipes)
    if total_recipes == 0:
        logging.warning("Daftar resep kosong. Tidak ada yang perlu dibagi.")
        return

    chunk_size = math.ceil(total_recipes / num_chunks)
    
    logging.info(f"Total resep: {total_recipes:,}")
    logging.info(f"Akan dibagi menjadi {num_chunks} file, masing-masing sekitar {chunk_size:,} resep.")

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min(start_index + chunk_size, total_recipes)
        
        current_chunk_data = all_recipes[start_index:end_index]
        
        if not current_chunk_data:
            continue

        name_without_ext = os.path.splitext(base_filename)[0]
        output_filename = f"{name_without_ext}-{i+1}.json"
        
        logging.info(f"Menulis chunk {i+1}/{num_chunks} ({len(current_chunk_data)} resep) ke: {output_filename}")
        
        with open(output_filename, 'w') as f:
            json.dump(current_chunk_data, f, indent=2)

    logging.info("Proses pembagian selesai.")

# --- Blok main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buat rencana kombinasi sampel untuk simulasi spektral.")
    
    parser.add_argument(
        "--num", 
        type=int, 
        default=10000,  # Default 10,000 sampel
        help="Jumlah total sampel kombinasi yang akan dibuat."
    )
    parser.add_argument(
        "--split", 
        type=int, 
        default=0,  # Default tidak dibagi
        help="Jumlah bagian file rencana akan dibagi. Jika 0 atau 1, tidak dibagi."
    )
    
    args = parser.parse_args()
    
    num_samples = args.num
    num_chunks_to_split = args.split
    
    file_suffix = num_samples // 1000
    output_filename = f"combinations-{file_suffix}.json"
    
    np.random.seed(42)  # Untuk reproduksibilitas
    
    all_combinations_data = generate_combinations_plan(num_samples, output_filename)

    if num_chunks_to_split > 1:
        logging.info(f"Memulai pembagian file menjadi {num_chunks_to_split} bagian.")
        split_json_plan(all_combinations_data, output_filename, num_chunks_to_split)
    elif num_chunks_to_split == 1:
        logging.info("Nilai --split adalah 1, tidak perlu pembagian file.")
    else:
        logging.info("Tidak ada pembagian file yang diminta.")