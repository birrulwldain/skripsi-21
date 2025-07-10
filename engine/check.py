import h5py
import numpy as np
import json
import os
from datetime import datetime
import argparse
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definisi elemen target (sesuaikan versi yang Anda gunakan)
# Versi 35
BASE_ELEMENTS = ["Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl", "Mg", "C", "S", "Ar", "Ti", "Mn", "Co"]
REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]

# Versi 9 (jika digunakan)
# BASE_ELEMENTS = ["Al", "Fe", "Ca", "Mg"]
# REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]

# --- Konfigurasi Statis ---
# Path dasar untuk dataset dan element map Anda.
# Sesuaikan path ini agar skrip bisa menemukannya.
BASE_DATASET_PATH_PREFIX = "/home/bwalidain/birrulwldain/dataset-"
ELEMENT_MAP_PATH = "/home/bwalidain/birrulwldain/element-map-18a.json"

STATIC_CONFIG = {
    "splits_to_check": ["train", "validation", "test"],
    "max_samples_per_split": 0, # 0 atau None berarti memproses SEMUA sampel di setiap split
    "results_dir": "reports" # Nama folder untuk laporan output
}

def load_element_map(element_map_path: str):
    """Memuat mapping elemen dari file JSON sebagai dictionary."""
    logging.info(f"Memuat element map dari: {element_map_path}")
    try:
        with open(element_map_path, 'r') as f:
            element_map = json.load(f)
        logging.info(f"Element map dimuat. Total {len(element_map)} kelas/ion.")
        
        # Verifikasi panjang nilai one-hot
        if not element_map:
            raise ValueError("Element map kosong.")
        first_one_hot_len = len(next(iter(element_map.values())))
        if not all(len(one_hot) == first_one_hot_len for one_hot in element_map.values()):
            logging.warning("Tidak semua nilai one-hot memiliki panjang yang sama dalam element map.")
        
        return element_map
    except FileNotFoundError:
        logging.error(f"Error: File element map tidak ditemukan di: {element_map_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error: Gagal mem-parsing file JSON element map: {e}")
        raise

def generate_combined_report(dataset_path: str, element_map_path: str, file_suffix: str):
    """
    Memeriksa proporsi kelas (elemen/ion) dalam dataset HDF5 untuk semua split
    dan menyimpan laporan gabungan ke satu file teks.
    """
    
    element_map = load_element_map(element_map_path)

    # Mapping indeks kelas ke nama kelas
    class_idx_to_name = {}
    for class_name, one_hot_vec in element_map.items():
        idx = np.argmax(one_hot_vec)
        class_idx_to_name[idx] = class_name
    
    # Ekstrak konfigurasi dari STATIC_CONFIG
    splits_to_check = STATIC_CONFIG["splits_to_check"]
    max_samples_per_split = STATIC_CONFIG["max_samples_per_split"]
    results_dir = STATIC_CONFIG["results_dir"]

    os.makedirs(results_dir, exist_ok=True) # Pastikan direktori laporan ada

    report_path = os.path.join(results_dir, f"report-dataset-{file_suffix}.txt")
    
    logging.info(f"Memulai pemeriksaan dataset: {dataset_path}")
    logging.info(f"Laporan gabungan akan disimpan ke: {report_path}")

    # Siapkan konten laporan gabungan
    combined_report_content = [
        f"Dataset Distribution Report (Generated: {datetime.now().isoformat()})\n",
        f"Dataset Path: {dataset_path}\n",
        f"Element Map Path: {element_map_path}\n\n"
    ]

    try:
        with h5py.File(dataset_path, 'r') as h5_file:
            for split in splits_to_check:
                if split not in h5_file:
                    logging.warning(f"Split '{split}' tidak ditemukan dalam {dataset_path}. Melewati.")
                    continue

                group = h5_file[split]
                
                # Gunakan slicing untuk membatasi jumlah sampel yang dibaca (jika max_samples_per_split > 0)
                # Jika max_samples_per_split adalah 0 atau None, ini akan membaca semua.
                labels_data = group['labels'][:] 
                if max_samples_per_split > 0:
                    labels_data = labels_data[:max_samples_per_split]
                
                logging.info(f"Memproses split: '{split}'. Menggunakan hingga {labels_data.shape[0]:,} sampel. Labels shape: {labels_data.shape}")

                # Hitung distribusi kelas di semua posisi
                num_classes = labels_data.shape[-1]
                all_labels_flat = labels_data.reshape(-1, num_classes) 
                
                class_counts = np.sum(all_labels_flat, axis=0)
                total_positions = all_labels_flat.shape[0]

                combined_report_content.append(f"--- Distribusi untuk SPLIT: '{split}' ---\n")
                combined_report_content.append(f"Total Sampel yang Diperiksa: {labels_data.shape[0]:,}\n")
                combined_report_content.append(f"Total Posisi Label: {total_positions:,}\n\n")
                combined_report_content.append("Class Distribution:\n")

                if total_positions == 0:
                    logging.warning(f"Split '{split}' memiliki 0 posisi. Tidak ada distribusi untuk dihitung.")
                    combined_report_content.append("  (Tidak ada data untuk split ini)\n\n")
                    continue

                # Simpan dan format distribusi per kelas
                current_split_distribution = {}
                for idx, count in enumerate(class_counts):
                    class_name = class_idx_to_name.get(idx, f"Class_{idx}_unknown")
                    proportion = (count / total_positions) * 100
                    current_split_distribution[class_name] = {"count": int(count), "proportion (%)": round(proportion, 4)}

                # Urutkan berdasarkan nama kelas untuk konsistensi dalam laporan
                sorted_classes = sorted(current_split_distribution.items())
                for class_name, stats in sorted_classes:
                    line = f"  {class_name}: {stats['count']:,} positions ({stats['proportion (%)']:.4f}%)\n"
                    logging.info(f"[Split {split}] {line.strip()}") # Cetak ke log juga
                    combined_report_content.append(line)
                combined_report_content.append("\n") # Baris kosong untuk pemisah antar split

    except FileNotFoundError:
        logging.critical(f"Error: Dataset HDF5 tidak ditemukan di: {dataset_path}. Pastikan path benar.")
        raise
    except Exception as e:
        logging.critical(f"Terjadi kesalahan tak terduga saat memproses dataset: {e}", exc_info=True)
        raise

    # Tulis seluruh konten laporan ke satu file
    with open(report_path, "w") as f:
        f.writelines(combined_report_content)
    logging.info(f"Laporan gabungan disimpan ke: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Periksa distribusi kelas dalam dataset HDF5 spektral dan buat laporan gabungan.")
    parser.add_argument(
        "--input", # Nama argumen baru
        type=str, 
        required=True, 
        help="Sufiks numerik dari nama file dataset (misal: '2' untuk dataset-2.h5)."
    )
    
    args = parser.parse_args()
    
    # Konstruksi dataset_path lengkap berdasarkan argumen --input
    # dan path prefix yang telah ditentukan
    full_dataset_path = f"{BASE_DATASET_PATH_PREFIX}{args.input}.h5"

    try:
        generate_combined_report(
            dataset_path=full_dataset_path,
            element_map_path=ELEMENT_MAP_PATH, # Diambil dari variabel global
            file_suffix=args.input # Gunakan input sebagai file_suffix
        )
    except Exception as e:
        logging.critical(f"Program berakhir dengan kesalahan fatal: {e}")
        exit(1)