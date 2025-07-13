

import torch
import h5py
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import logging
import pandas as pd # Added pandas import
from train import InformerModel, set_seed
from scipy.signal import find_peaks

def analyze_from_h5(model, element_map, config, sample_idx, split, peak_height, prominence, distance, threshold):
    print(f"\n--- Menganalisis Sampel #{sample_idx} dari Set '{split}' (H5) ---")
    h5_path, class_names = config["paths"]["h5_dataset_path"], list(element_map.keys())
    device = next(model.parameters()).device
    with h5py.File(h5_path, 'r') as f:
        spectrum, ground_truth, wavelengths = (f[split]['spectra'][sample_idx], f[split]['labels'][sample_idx], f['wavelengths'][:])
    input_tensor = torch.from_numpy(spectrum[np.newaxis, :, np.newaxis]).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    predictions = (probabilities > threshold).astype(int)

    # Call static plot function
    base_output_path = os.path.join(os.path.dirname(__file__), '..', 'images', f'prediction_plot_sample_{sample_idx}.png')
    create_static_plot(wavelengths, spectrum, predictions, probabilities, class_names, ground_truth, peak_height, prominence, distance, title_prefix=f"Perbandingan Prediksi vs. Ground Truth (Sampel #{sample_idx})", base_output_path=base_output_path)


def create_static_plot(
    wavelengths: np.ndarray, spectrum: np.ndarray, predictions: np.ndarray,
    probabilities: np.ndarray, class_names: list, ground_truth: np.ndarray = None,
    peak_height: float = 0.05, prominence: float = None, distance: float = 8, # Added prominence, distance
    title_prefix: str = "Visualisasi Prediksi Elemen", base_output_path: str = None # Changed title, output_path
):
    """
    Fungsi untuk membuat plot statis menggunakan Matplotlib, dibagi menjadi dua rentang panjang gelombang.
    """
    logger = logging.getLogger('AnalysisToolkit')
    logger.info("Membuat plot statis dengan Matplotlib...")

    ranges = [(200, 375), (375, 550), (550, 725), (725, 900)]
    range_titles = ["200nm - 375nm", "375nm - 550nm", "550nm - 725nm", "725nm - 900nm"]

    for i, (wl_min, wl_max) in enumerate(ranges):
        # Filter data for the current wavelength range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        current_wavelengths = wavelengths[mask]
        current_spectrum = spectrum[mask]
        current_predictions = predictions[mask]
        current_probabilities = probabilities[mask]
        current_ground_truth = ground_truth[mask] if ground_truth is not None else None

        if len(current_wavelengths) == 0:
            logger.warning(f"Tidak ada data untuk rentang {wl_min}-{wl_max}nm. Melewatkan plot ini.")
            continue

        fig, ax = plt.subplots(figsize=(18, 9)) # New figure for each range

        # 1. Plot spektrum utama
        ax.plot(current_wavelengths, current_spectrum, color='gray', label='Spektrum Input', linewidth=1.2, zorder=1)

        # 2. Temukan dan tandai puncak (using dynamic peak_height, prominence, distance)
        peak_indices, _ = find_peaks(current_spectrum, height=peak_height, distance=distance, prominence=prominence)
        ax.plot(current_wavelengths[peak_indices], current_spectrum[peak_indices], 'x', color='red', markersize=8, label='Puncak Terdeteksi', zorder=2)

        # 3. Tambahkan anotasi untuk setiap puncak (original annotation logic)
        for peak_idx in peak_indices: # Loop through peak_indices for current range
            peak_wl = current_wavelengths[peak_idx]
            peak_int = current_spectrum[peak_idx]

            # Kumpulkan teks anotasi
            pred_labels = []
            pred_at_peak = current_predictions[peak_idx]
            detected_indices = np.where(pred_at_peak == 1)[0]
            for j in detected_indices:
                if class_names[j] != 'background':
                    pred_labels.append(class_names[j])

            # Bangun string anotasi final
            annotation_text = ""
            if pred_labels:
                annotation_text += ", ".join(pred_labels)

            # Tambahkan anotasi ke plot jika ada teks
            if annotation_text:
                ax.annotate(
                    annotation_text,
                    xy=(peak_wl, peak_int),
                    xytext=(peak_wl, peak_int + 0.08),
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black')
                )

        # 4. Atur properti plot
        ax.set_title(f"{title_prefix} ({range_titles[i]})", fontsize=16)
        ax.set_xlabel("Panjang Gelombang (nm)", fontsize=12)
        ax.set_ylabel("Intensitas Normalisasi", fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(bottom=0, top=max(1.0, current_spectrum.max() * 1.1))
        ax.set_xlim(left=wl_min, right=wl_max)

        plt.tight_layout()
        if base_output_path:
            output_filename = f"{base_output_path.replace('.png', '')}_range_{i+1}.png"
            plt.savefig(output_filename, dpi=300)
            logger.info(f"Plot disimpan di: {output_filename}")
        plt.show()



def save_prediction_details_to_txt(
    wavelengths: np.ndarray, spectrum: np.ndarray, predictions: np.ndarray,
    probabilities: np.ndarray, class_names: list, ground_truth: np.ndarray,
    peak_indices: np.ndarray, output_filename: str = "prediction_details.txt"
):
    logger = logging.getLogger('AnalysisToolkit')
    logger.info(f"Menyimpan detail prediksi ke {output_filename}...")

    with open(output_filename, 'w') as f:
        # Header
        header = ["Panjang Gelombang (nm)", "Intensitas", "Prediksi Atom (Probabilitas)", "Atom Asli"]
        f.write("\t".join(header) + "\n")

        # Sort peaks by wavelength
        sorted_peak_indices = peak_indices[np.argsort(wavelengths[peak_indices])]

        for peak_idx in sorted_peak_indices:
            peak_wl = wavelengths[peak_idx]
            peak_int = spectrum[peak_idx]

            # Predicted atoms with probabilities
            pred_labels_with_prob = []
            pred_at_peak = predictions[peak_idx]
            detected_indices = np.where(pred_at_peak == 1)[0]
            for j in detected_indices:
                if class_names[j] != 'background':
                    prob_percent = probabilities[peak_idx, j] * 100
                    pred_labels_with_prob.append(f"{class_names[j]} ({prob_percent:.1f}%)")
            predicted_str = ", ".join(pred_labels_with_prob) if pred_labels_with_prob else "Tidak Ada"

            # True atoms
            gt_labels = []
            if ground_truth is not None:
                gt_at_peak = ground_truth[peak_idx]
                true_indices = np.where(gt_at_peak == 1)[0]
                for j in true_indices:
                    if class_names[j] != 'background':
                        gt_labels.append(class_names[j])
            true_str = ", ".join(gt_labels) if gt_labels else "Tidak Ada"

            f.write(f"{peak_wl:.3f}\t{peak_int:.3f}\t{predicted_str}\t{true_str}\n")
    logger.info(f"Detail prediksi berhasil disimpan di {output_filename}")

def analyze_from_asc(model, element_map, config, asc_file_path, peak_height, prominence, distance, threshold):
    print(f"\n--- Menganalisis Data Lapangan dari File: {os.path.basename(asc_file_path)} ---")
    def _prepare_asc_file(asc_path, h5_grid_path):
        df = pd.read_csv(asc_path, sep=r'\s+', names=['wavelength', 'intensity'], comment='#')
        original_wl, original_spec = df['wavelength'].values, df['intensity'].values
        with h5py.File(h5_grid_path, 'r') as f: target_wl = f['wavelengths'][:]
        baseline = np.percentile(original_spec, 5)
        spec_corrected = original_spec - baseline
        spec_corrected[spec_corrected < 0] = 0
        max_val = np.max(spec_corrected)
        spec_normalized = (spec_corrected / max_val) * 0.8 if max_val > 0 else spec_corrected
        return target_wl, np.interp(target_wl, original_wl, spec_normalized).astype(np.float32)
    try:
        wavelengths, spectrum = _prepare_asc_file(asc_file_path, config["paths"]["h5_dataset_path"])
    except Exception as e:
        print(f"Gagal memproses file .asc: {e}")
        return
    class_names, device = list(element_map.keys()), next(model.parameters()).device
    input_tensor = torch.from_numpy(spectrum[np.newaxis, :, np.newaxis]).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    predictions = (probabilities > threshold).astype(int)

    # Calculate peak_indices here for ASC data
    peak_indices, _ = find_peaks(spectrum, height=peak_height, distance=distance, prominence=prominence)

    # Call static plot function
    base_output_path = os.path.join(os.path.dirname(__file__), '..', 'images', f'prediction_plot_{os.path.basename(asc_file_path).replace('.asc', '')}.png')
    create_static_plot(wavelengths, spectrum, predictions, probabilities, class_names, None, peak_height, prominence, distance, title_prefix=f"Prediksi pada Data Lapangan ({os.path.basename(asc_file_path)})", base_output_path=base_output_path)

    # Save prediction details to text file for ASC data
    output_filename = os.path.join(os.path.dirname(__file__), '..', 'images', f'prediction_details_{os.path.basename(asc_file_path).replace('.asc', '')}.txt')
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    save_prediction_details_to_txt(wavelengths, spectrum, predictions, probabilities, class_names, None, peak_indices, output_filename=output_filename)


def load_model(model_path, model_config, device):
    """Memuat model Informer dari file .pth."""
    model = InformerModel(**model_config)
    state_dict = torch.load(model_path, map_location=device)
    
    # Hapus awalan '_orig_mod.' jika ada
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[10:] if k.startswith('_orig_mod.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def get_spectrum_sample(dataset_path, index=0):
    """Mengambil satu sampel spektrum dan label dari test set."""
    with h5py.File(dataset_path, 'r') as f:
        spectrum = f['test']['spectra'][index]
        labels = f['test']['labels'][index]
    return spectrum, labels

def predict(model, spectrum, device):
    """Melakukan prediksi pada satu sampel spektrum."""
    spectrum_tensor = torch.from_numpy(spectrum[np.newaxis, ..., np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        output = model(spectrum_tensor)
        probs = torch.sigmoid(output)
    return probs.squeeze().cpu().numpy()



if __name__ == '__main__':
    set_seed(42)

    # Konfigurasi
    config = {
        "paths": {
            "h5_dataset_path": '/Volumes/Private-C/Private-C-backup/skripsi-21/engine/dataset-50.h5',
            "element_map_path": '/Volumes/Private-C/Private-C-backup/skripsi-21/engine/element-map-18a.json',
            "model_path": '/Volumes/Private-C/Private-C-backup/skripsi-21/engine/data/informer_multilabel_model-2.pth',
            "asc_data_dir": '/Volumes/Private-C/Private-C-backup/skripsi-21/engine/data/telur/' # New: Directory for .asc files
        },
        "model_params": {
            "input_dim": 1,
            "d_model": 32,
            "nhead": 4,
            "num_encoder_layers": 2,
            "dim_feedforward": 64,
            "dropout": 0.4,
            "seq_length": 4096,
            "attn_factor": 5
        },
        "analysis_params": {
            "sample_idx": 10, # Ganti index untuk sampel lain
            "split": "test",
            "peak_height": 0.001, # Sesuaikan jika perlu
            "prominence": 0.004, # Tambahkan jika perlu
            "distance": 1, # Tambahkan jika perlu
            "threshold": 0.7 # Sesuaikan jika perlu
        }
    }

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(config["paths"]["element_map_path"], 'r') as f:
        element_map = json.load(f)

    config["model_params"]["num_classes"] = len(element_map)

    # Muat model
    model = load_model(config["paths"]["model_path"], config["model_params"], DEVICE)

    # Analisis dan plot hasil dari H5 dataset
    analyze_from_h5(
        model,
        element_map,
        config,
        config["analysis_params"]["sample_idx"],
        config["analysis_params"]["split"],
        config["analysis_params"]["peak_height"],
        config["analysis_params"]["prominence"],
        config["analysis_params"]["distance"],
        config["analysis_params"]["threshold"]
    )

    # Analisis dan plot hasil dari file .asc
    # Ganti 'sample.asc' dengan nama file .asc yang ingin Anda analisis
    asc_file_to_analyze = os.path.join(config["paths"]["asc_data_dir"], 'Grup-1.asc') # Placeholder
    if os.path.exists(asc_file_to_analyze):
        analyze_from_asc(
            model,
            element_map,
            config,
            asc_file_to_analyze,
            config["analysis_params"]["peak_height"],
            config["analysis_params"]["prominence"],
            config["analysis_params"]["distance"],
            config["analysis_params"]["threshold"]
        )
    else:
        print(f"Peringatan: File .asc tidak ditemukan di {asc_file_to_analyze}. Melewatkan analisis .asc.")

