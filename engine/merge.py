# Nama file: merger.py (Lengkap - Multi-Label & Atom-Only, Koreksi NameError)
import h5py
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict
import math
import logging
from datetime import datetime
import pandas as pd # Digunakan untuk pd.Timestamp.now().isoformat()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfigurasi Tambahan untuk Pelabelan Atom-Only ---
# Path ke element map asli yang berisi spesies ion (misal: "Al_1", "Al_2")
ORIGINAL_ELEMENT_MAP_PATH = "/home/bwalidain/birrulwldain/element-map-35.json"
# Path untuk menyimpan element map baru yang hanya berisi elemen dasar + background
NEW_ELEMENT_MAP_ATOM_ONLY_PATH = "/home/bwalidain/birrulwldain/element-map-18a.json" # Gunakan nama yang sudah ada

def load_original_element_map(map_path: str) -> dict:
    """Memuat element map asli (spesies ion) dari file JSON."""
    try:
        with open(map_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.critical(f"Error: Original element map tidak ditemukan di {map_path}.")
        raise
    except json.JSONDecodeError as e:
        logging.critical(f"Error: Gagal mem-parsing element map asli di {map_path}: {e}")
        raise

def create_atom_only_element_map(original_element_map: dict, new_map_path: str) -> dict:
    """
    Membuat element map baru yang hanya memetakan elemen dasar (atom) dan background
    ke vektor one-hot yang sesuai.
    """
    logging.info("Membuat element map baru (atom-only)...")
    
    # Ekstrak semua elemen dasar dan pastikan 'background' ada
    base_elements = sorted(list(set([k.split('_')[0] for k in original_element_map.keys() if k != 'background'])))
    all_target_classes = base_elements + ['background']
    num_atom_classes = len(all_target_classes)
    
    atom_only_map = {}
    for i, class_name in enumerate(all_target_classes):
        one_hot = [0] * num_atom_classes
        one_hot[i] = 1
        atom_only_map[class_name] = one_hot
        
    try:
        with open(new_map_path, 'w') as f:
            json.dump(atom_only_map, f, indent=2)
        logging.info(f"Element map atom-only disimpan ke: {new_map_path} dengan {num_atom_classes} kelas.")
    except Exception as e:
        logging.error(f"Gagal menyimpan element map atom-only: {e}")
        raise

    return atom_only_map, num_atom_classes

def merge_and_split_manually(chunk_files: list, final_output_path: str):
    """
    Menggabungkan HDF5 chunks, mengonversi label ke elemen dasar,
    dan melakukan STRATIFIED SPLIT.
    """
    logging.info(f"Memulai penggabungan {len(chunk_files)} file chunk ke {final_output_path}")

    # Muat element map asli dan buat yang baru untuk atom-only
    original_element_map = load_original_element_map(ORIGINAL_ELEMENT_MAP_PATH)
    atom_only_element_map, num_atom_classes = create_atom_only_element_map(original_element_map, NEW_ELEMENT_MAP_ATOM_ONLY_PATH)

    # Buat mapping dari indeks one-hot asli ke nama spesies ion
    original_idx_to_ion_name = {np.argmax(v): k for k, v in original_element_map.items()}

    # Buat mapping dari nama spesies ion ke nama elemen dasar
    ion_name_to_atom_name = {k: k.split('_')[0] for k in original_element_map.keys() if k != 'background'}
    ion_name_to_atom_name['background'] = 'background'

    all_spectra = []
    all_original_labels = [] # Tetap baca label asli dari chunk
    all_atom_percentages_str = []
    wavelengths = None

    # 1. Baca dan gabungkan semua data
    for file_path in tqdm(chunk_files, desc="Membaca semua chunk"):
        try:
            with h5py.File(file_path, 'r') as f:
                if 'chunk_data' not in f:
                    logging.warning(f"Grup 'chunk_data' tidak ditemukan di {file_path}. Melewati file ini.")
                    continue
                
                all_spectra.append(f['chunk_data/spectra'][:])
                all_original_labels.append(f['chunk_data/labels'][:]) # Label asli dari job.py (seharusnya multi-hot ion)
                all_atom_percentages_str.append(f['chunk_data/atom_percentages'][:])
                
                if wavelengths is None and 'wavelengths' in f:
                    wavelengths = f['wavelengths'][:]
                elif wavelengths is None:
                    logging.warning(f"Kunci 'wavelengths' tidak ditemukan di {file_path}. Pastikan setidaknya satu chunk memilikinya.")

        except Exception as e:
            logging.error(f"Gagal membaca file {file_path}: {e}")
            continue
    
    if not all_spectra:
        raise ValueError("Tidak ada data valid yang berhasil dibaca dari file-file chunk. Proses dibatalkan.")
    if wavelengths is None:
        raise ValueError("Wavelengths tidak ditemukan di chunk manapun. Tidak dapat melanjutkan.")

    logging.info("Menggabungkan data ke dalam memori...")
    final_spectra = np.concatenate(all_spectra, axis=0)
    final_original_labels = np.concatenate(all_original_labels, axis=0) # Label asli gabungan (multi-hot ion)
    final_atom_percentages_str = np.concatenate(all_atom_percentages_str, axis=0)
    total_samples = len(final_spectra)
    logging.info(f"Total sampel setelah digabung: {total_samples:,}")

    # --- KOREKSI: Konversi Label dari Spesies Ion (Multi-Hot) ke Elemen Dasar (Multi-Hot) ---
    logging.info("Mengonversi label dari spesies ion (multi-hot) ke elemen dasar (atom-only, multi-hot)...")
    
    # Dapatkan original_num_classes dari shape label yang digabungkan
    original_num_classes = final_original_labels.shape[-1]
    
    # final_atom_labels akan menjadi array label multi-hot atom-only
    final_atom_labels = np.zeros((final_original_labels.shape[0], final_original_labels.shape[1], num_atom_classes), dtype=np.float32)
    
    # Dapatkan indeks untuk label background di map atom-only
    atom_only_background_idx = np.argmax(atom_only_element_map['background'])

    for sample_idx in tqdm(range(final_original_labels.shape[0]), desc="Konversi Multi-Label"):
        for pixel_idx in range(final_original_labels.shape[1]):
            original_multi_hot = final_original_labels[sample_idx, pixel_idx, :]
            
            current_pixel_atom_labels = np.zeros(num_atom_classes, dtype=np.float32)
            elements_active_in_pixel = False

            # Tahap 1: Identifikasi semua elemen yang kontribusinya di atas multi_label_contribution_threshold
            # dan masukkan mereka sebagai kandidat label multi-hot.
            # Loop melalui setiap kemungkinan ion di label asli (35 kelas)
            for original_idx_in_one_hot in range(original_num_classes): # Gunakan original_num_classes yang sudah didefinisikan
                # Jika bit ini aktif di label asli (berarti ion ini ada di piksel ini)
                if original_multi_hot[original_idx_in_one_hot] == 1:
                    ion_name = original_idx_to_ion_name.get(original_idx_in_one_hot)
                    
                    if ion_name and ion_name != 'background': # Jika ini adalah ion, bukan background asli
                        atom_name = ion_name_to_atom_name.get(ion_name) # Dapatkan nama elemen dasar (e.g., "Fe" dari "Fe_1")
                        if atom_name and atom_name in atom_only_element_map:
                            atom_one_hot = np.array(atom_only_element_map[atom_name], dtype=np.float32)
                            current_pixel_atom_labels += atom_one_hot # Tambahkan kontribusi one-hot atom
                            elements_active_in_pixel = True
                    # Jika yang aktif di original_multi_hot adalah background
                    elif ion_name == 'background':
                        # Untuk multi-label, background hanya aktif jika TIDAK ADA elemen lain yang aktif
                        pass # Akan ditangani di Tahap 2
                        
            # Tahap 2: Tentukan label akhir untuk piksel ini
            if elements_active_in_pixel:
                # Jika ada satu atau lebih elemen atom aktif yang ditemukan,
                # buat label multi-hot untuk semua elemen tersebut dan pastikan background MATI.
                current_pixel_atom_labels = (current_pixel_atom_labels > 0).astype(np.float32) # Normalisasi ke biner
                current_pixel_atom_labels[atom_only_background_idx] = 0 # Pastikan background nonaktif
            else:
                # Jika tidak ada elemen atom aktif yang ditemukan di atas ambang batas,
                # maka piksel ini adalah background.
                current_pixel_atom_labels = np.zeros(num_atom_classes, dtype=np.float32) # Set semua ke 0
                current_pixel_atom_labels[atom_only_background_idx] = 1 # Set background aktif

            # KOREKSI DISINI: Simpan ke final_atom_labels
            final_atom_labels[sample_idx, pixel_idx, :] = current_pixel_atom_labels
    
    final_labels_for_split = final_atom_labels

    # 2. Buat kunci stratifikasi (tetap berdasarkan elemen dasar dominan)
    logging.info("Membuat kunci stratifikasi untuk pembagian final (berbasis elemen dasar)...")
    strata_keys = []
    for json_str in tqdm(final_atom_percentages_str, desc="Parsing metadata gabungan untuk stratifikasi"):
        try:
            data = json.loads(json_str.decode('utf-8'))
            element_percentages = {k: v for k, v in data.items() if k not in ['temperature', 'electron_density', 'delta_E_max', 'sample_id']}
            
            # Agregasi persentase ion ke elemen dasar
            atom_percentages_aggr = defaultdict(float)
            for ion_key, percentage in element_percentages.items():
                atom_name = ion_key.split('_')[0] if ion_key != 'background' else 'background'
                atom_percentages_aggr[atom_name] += percentage
            
            dominant_element_atom = max(atom_percentages_aggr, key=atom_percentages_aggr.get) if atom_percentages_aggr else 'none'
            strata_keys.append(dominant_element_atom)
        except json.JSONDecodeError as e:
            logging.error(f"Gagal mem-parsing JSON dari string: {json_str.decode('utf-8')[:100]}... Error: {e}. Menggunakan 'corrupted_json' sebagai kunci.")
            strata_keys.append('corrupted_json')

    logging.info("Distribusi sampel untuk stratifikasi (berbasis elemen dasar):")
    class_distribution = Counter(strata_keys)
    for key, count in sorted(class_distribution.items(), key=lambda item: item[1]):
        logging.info(f"  - {key}: {count} sampel")
    
    # --- LOGIKA PEMBAGIAN MANUAL SEPENUHNYA ---
    logging.info("Melakukan pembagian manual yang tangguh untuk setiap kelas (berbasis elemen dasar)...")
    
    strata_groups = defaultdict(list)
    for i, key in enumerate(strata_keys):
        strata_groups[key].append(i)

    final_train_idx, final_val_idx, final_test_idx = [], [], []
    
    train_ratio, val_ratio = 0.7, 0.15

    for key, indices in tqdm(strata_groups.items(), desc="Membagi setiap kelas"):
        n_samples_in_class = len(indices)
        np.random.shuffle(indices)

        if n_samples_in_class < 3:
            final_train_idx.extend(indices)
            logging.info(f"  - Kelas '{key}': Terlalu kecil ({n_samples_in_class} sampel). Semua ke training set.")
            continue

        n_train = math.floor(train_ratio * n_samples_in_class)
        n_val = math.floor(val_ratio * n_samples_in_class)

        if n_train == 0 and n_samples_in_class > 0: n_train = 1
        if n_val == 0 and n_samples_in_class - n_train > 0: n_val = 1
        n_test = max(0, n_samples_in_class - n_train - n_val)

        current_idx = 0
        train_slice = indices[current_idx : current_idx + n_train]
        current_idx += len(train_slice)
        
        val_slice = indices[current_idx : current_idx + n_val]
        current_idx += len(val_slice)
        
        test_slice = indices[current_idx:]

        final_train_idx.extend(train_slice)
        final_val_idx.extend(val_slice)
        final_test_idx.extend(test_slice)
        
    np.random.shuffle(final_train_idx)
    np.random.shuffle(final_val_idx)
    np.random.shuffle(final_test_idx)

    logging.info(f"Ukuran final (stratified): Train={len(final_train_idx):,}, Validation={len(final_val_idx):,}, Test={len(final_test_idx):,}")
    
    total_split_samples = len(final_train_idx) + len(final_val_idx) + len(final_test_idx)
    if total_split_samples != total_samples:
        logging.warning(f"Ketidakcocokan jumlah sampel setelah split: Asli={total_samples}, Split={total_split_samples}. Mungkin ada pembulatan atau data yang tidak terdistribusi.")

    logging.info(f"\nMenyimpan dataset final ke {final_output_path}...")
    with h5py.File(final_output_path, 'w') as f:
        f.create_dataset('wavelengths', data=wavelengths, compression='gzip', compression_opts=9)

        def write_final_group(group_name, indices, labels_to_use):
            grp = f.create_group(group_name)
            grp.create_dataset('spectra', data=final_spectra[indices], compression='gzip', compression_opts=9)
            grp.create_dataset('labels', data=labels_to_use[indices], compression='gzip', compression_opts=9)
            grp.create_dataset('atom_percentages', data=final_atom_percentages_str[indices], compression='gzip', compression_opts=9)
            grp.attrs['num_samples'] = len(indices)
        
        write_final_group('train', final_train_idx, final_labels_for_split)
        write_final_group('validation', final_val_idx, final_labels_for_split)
        write_final_group('test', final_test_idx, final_labels_for_split)
        
        f.attrs['total_samples_merged'] = total_samples
        f.attrs['train_samples'] = len(final_train_idx)
        f.attrs['validation_samples'] = len(final_val_idx)
        f.attrs['test_samples'] = len(final_test_idx)
        f.attrs['train_ratio'] = train_ratio
        f.attrs['validation_ratio'] = val_ratio
        f.attrs['test_ratio'] = 1 - train_ratio - val_ratio
        f.attrs['source_files'] = json.dumps([os.path.basename(p) for p in chunk_files])
        f.attrs['merge_timestamp'] = pd.Timestamp.now().isoformat()
        f.attrs['description'] = "HDF5 dataset hasil penggabungan dan stratified split berdasarkan elemen dominan atom. Label piksel dikonversi ke elemen dasar (multi-hot)."
        f.attrs['num_atom_classes'] = num_atom_classes

    logging.info("\nPenggabungan dan stratifikasi final selesai! ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Menggabungkan HDF5 Chunks, Mengonversi Label ke Atom Dasar (Multi-Hot), dan Melakukan Stratified Split")
    parser.add_argument(
        '--inputs', 
        type=str, 
        nargs='+',
        required=True, 
        help='Daftar path file HDF5 chunk (bisa pakai wildcard, contoh: "dataset-2-*.h5")'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='Path untuk file HDF5 final (contoh: "dataset-2.h5")'
    )
    
    args = parser.parse_args()
    
    existing_chunk_files = [f for f in args.inputs if os.path.exists(f)]

    if not existing_chunk_files:
        logging.error("Tidak ada file input yang valid ditemukan. Pastikan path dan wildcard benar.")
    else:
        np.random.seed(42)
        try:
            merge_and_split_manually(existing_chunk_files, args.output)
        except Exception as e:
            logging.critical(f"Terjadi kesalahan fatal selama proses merge, konversi (multi-hot), dan split: {e}", exc_info=True)