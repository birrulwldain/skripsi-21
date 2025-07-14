# =============================================================================
# BAGIAN 1: IMPOR, KONFIGURASI, DAN DEFINISI MODEL
# =============================================================================

# --- Impor Pustaka Utama ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import numpy as np
import pandas as pd
import json
import os
import math
import random
from datetime import datetime
import argparse
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, f1_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
from scipy.signal import find_peaks
import traceback
import logging
from typing import Dict, Tuple, List

# --- Konfigurasi Analisis (Default, bisa di-override oleh argumen) ---
CONFIG = {
    "model": {
        "input_dim": 1, "d_model": 32, "nhead": 4, "num_encoder_layers": 3,
        "dim_feedforward": 64, "dropout": 0.5, "seq_length": 4096,
        "attn_factor": 5, "num_classes": 18
    }
}

# --- Definisi Kelas-kelas Model (Wajib Ada untuk Memuat Bobot) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, attn_factor):
        super(ProbSparseSelfAttention, self).__init__()
        self.d_model, self.nhead, self.d_k, self.factor = d_model, nhead, d_model // nhead, attn_factor
        self.q_linear, self.k_linear, self.v_linear, self.out_linear = (nn.Linear(d_model, d_model) for _ in range(4))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, L, _ = x.shape; H, D = self.nhead, self.d_k
        Q, K, V = (l(x).view(B, L, H, D).transpose(1, 2) for l in (self.q_linear, self.k_linear, self.v_linear))
        U = min(L, int(self.factor * math.log(L)) if L > 1 else L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None: scores.masked_fill_(mask == 0, -float('inf'))
        top_k, _ = torch.topk(scores, U, dim=-1)
        scores.masked_fill_(scores < top_k[..., -1, None], -float('inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_linear(context)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, attn_factor):
        super(EncoderLayer, self).__init__()
        self.self_attention = ProbSparseSelfAttention(d_model, nhead, dropout, attn_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.self_attention(x, mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x

class InformerModel(nn.Module):
    def __init__(self, **kwargs):
        super(InformerModel, self).__init__()
        self.d_model = kwargs["d_model"]
        self.embedding = nn.Linear(kwargs["input_dim"], self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, kwargs["seq_length"])
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model, nhead=kwargs["nhead"], dim_feedforward=kwargs["dim_feedforward"],
                dropout=kwargs["dropout"], attn_factor=kwargs["attn_factor"]
            ) for _ in range(kwargs["num_encoder_layers"])
        ])
        self.decoder = nn.Linear(self.d_model, kwargs["num_classes"])
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.decoder(x)

# =============================================================================
# BAGIAN 2: KELAS DAN FUNGSI HELPER UNTUK ANALISIS
# =============================================================================

class HDF5SpectrumDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, split, max_samples=None):
        self.file_path, self.split, self.h5_file = file_path, split, None
        with h5py.File(self.file_path, 'r') as f:
            if split not in f: raise KeyError(f"Split '{split}' tidak ada di {file_path}")
            total_len = len(f[split]['spectra'])
            self.dataset_len = min(total_len, max_samples) if max_samples is not None else total_len
    def __len__(self): return self.dataset_len
    def __getitem__(self, idx):
        if self.h5_file is None: self.h5_file = h5py.File(self.file_path, 'r')
        spectra = self.h5_file[self.split]['spectra'][idx][..., np.newaxis].astype(np.float32)
        labels = self.h5_file[self.split]['labels'][idx].astype(np.float32)
        return torch.from_numpy(spectra), torch.from_numpy(labels)

def setup_logging():
    logger = logging.getLogger('AnalysisToolkit')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

def load_analysis_tools(model_path: str, element_map_path: str, model_config: dict) -> Tuple[nn.Module, dict]:
    logger = logging.getLogger('AnalysisToolkit')
    logger.info("Memuat model dan peta elemen...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Perbarui jumlah kelas dari peta elemen
    with open(element_map_path, 'r') as f:
        element_map = json.load(f)
    model_config['num_classes'] = len(element_map)

    model = InformerModel(**model_config).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    # Membersihkan state_dict jika model disimpan setelah torch.compile()
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    
    logger.info(f"Model berhasil dimuat dari {model_path} ke {device.upper()}.")
    return model, element_map

def create_static_plot(
    wavelengths: np.ndarray, spectrum: np.ndarray, predictions: np.ndarray,
    probabilities: np.ndarray, class_names: list, output_path: str, ground_truth: np.ndarray = None,
    peak_height: float = 0.05, title: str = "Visualisasi Prediksi Elemen"
):
    logger = logging.getLogger('AnalysisToolkit')
    logger.info(f"Membuat plot statis dan menyimpannya ke: {output_path}")
    fig, ax = plt.subplots(figsize=(18, 9))

    ax.plot(wavelengths, spectrum, color='gray', label='Spektrum Input', linewidth=1.2, zorder=1)
    peak_indices, _ = find_peaks(spectrum, height=peak_height, distance=8)
    ax.plot(wavelengths[peak_indices], spectrum[peak_indices], 'x', color='red', markersize=8, label='Puncak Terdeteksi', zorder=2)

    for i, peak_idx in enumerate(peak_indices):
        peak_wl, peak_int = wavelengths[peak_idx], spectrum[peak_idx]
        pred_labels = [f"{class_names[j]} ({probabilities[peak_idx, j]*100:.1f}%)" 
                       for j in np.where(predictions[peak_idx] == 1)[0] if class_names[j] != 'background']
        gt_labels = []
        if ground_truth is not None:
            gt_labels = [class_names[j] for j in np.where(ground_truth[peak_idx] == 1)[0] if class_names[j] != 'background']

        annotation_text = ""
        if pred_labels: annotation_text += "Pred: " + ", ".join(pred_labels)
        if gt_labels: annotation_text += "\nAsli: " + ", ".join(gt_labels)

        if annotation_text:
            ax.annotate(
                annotation_text, xy=(peak_wl, peak_int), xytext=(peak_wl, peak_int + 0.08),
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black')
            )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Panjang Gelombang (nm)", fontsize=12)
    ax.set_ylabel("Intensitas Normalisasi", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0, top=max(1.0, spectrum.max() * 1.1))
    ax.set_xlim(left=wavelengths.min(), right=wavelengths.max())

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info("Plot berhasil disimpan.")

# =============================================================================
# BAGIAN 3: FUNGSI-FUNGSI ANALISIS UTAMA
# =============================================================================

def run_final_evaluation(model, element_map, dataset_path, results_dir, test_split_name='test', threshold=0.5):
    logger = logging.getLogger('AnalysisToolkit')
    logger.info(f"\n--- MEMULAI EVALUASI KUANTITATIF FINAL PADA SET '{test_split_name.upper()}' ---")
    device = next(model.parameters()).device
    num_classes = len(element_map)
    
    try:
        test_dataset = HDF5SpectrumDataset(file_path=dataset_path, split=test_split_name)
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)
    except KeyError:
        logger.error(f"Split '{test_split_name}' tidak ditemukan di {dataset_path}. Evaluasi dibatalkan.")
        return

    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc=f"Mengevaluasi pada set {test_split_name}"):
            data, target = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(data)
            all_preds.append((torch.sigmoid(output) > threshold).int().cpu())
            all_targets.append(target.cpu())

    y_pred = torch.cat(all_preds).view(-1, num_classes).numpy()
    y_true = torch.cat(all_targets).view(-1, num_classes).numpy()

    logger.info("\nMenghitung metrik performa...")
    class_names = list(element_map.keys())
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, digits=4)
    
    report_path = os.path.join(results_dir, f"evaluation_report_{test_split_name}.txt")
    with open(report_path, 'w') as f:
        f.write(f"EVALUASI PADA SPLIT: {test_split_name.upper()}\n")
        f.write(f"AMBANG BATAS PREDIKSI: {threshold}\n")
        f.write("="*50 + "\n")
        f.write(report_str)
    
    logger.info(f"Laporan evaluasi disimpan di: {report_path}")
    print("\n" + report_str)


def analyze_from_h5(model, element_map, dataset_path, results_dir, sample_idx, split, peak_height, threshold):
    logger = logging.getLogger('AnalysisToolkit')
    logger.info(f"\n--- Menganalisis Sampel #{sample_idx} dari Set '{split}' (H5) ---")
    class_names = list(element_map.keys())
    device = next(model.parameters()).device
    
    with h5py.File(dataset_path, 'r') as f:
        spectrum = f[split]['spectra'][sample_idx]
        ground_truth = f[split]['labels'][sample_idx]
        wavelengths = f['wavelengths'][:]
        
    input_tensor = torch.from_numpy(spectrum[np.newaxis, :, np.newaxis]).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    predictions = (probabilities > threshold).astype(int)
    
    output_filename = f"h5_sample_{sample_idx}_{split}_visualization.png"
    output_path = os.path.join(results_dir, output_filename)
    
    create_static_plot(
        wavelengths, spectrum, predictions, probabilities, class_names, output_path,
        ground_truth=ground_truth, peak_height=peak_height,
        title=f"Perbandingan Prediksi vs. Ground Truth (Sampel #{sample_idx} dari split '{split}')"
    )

def analyze_from_asc(model, element_map, dataset_path, results_dir, asc_file_path, peak_height, threshold):
    logger = logging.getLogger('AnalysisToolkit')
    logger.info(f"\n--- Menganalisis Data Lapangan dari File: {os.path.basename(asc_file_path)} ---")
    
    def _prepare_asc_file(asc_path, h5_grid_path):
        df = pd.read_csv(asc_path, sep='\s+', names=['wavelength', 'intensity'], comment='#')
        original_wl, original_spec = df['wavelength'].values, df['intensity'].values
        with h5py.File(h5_grid_path, 'r') as f: target_wl = f['wavelengths'][:]
        baseline = np.percentile(original_spec, 5)
        spec_corrected = original_spec - baseline
        spec_corrected[spec_corrected < 0] = 0
        max_val = np.max(spec_corrected)
        spec_normalized = (spec_corrected / max_val) * 0.8 if max_val > 0 else spec_corrected
        return target_wl, np.interp(target_wl, original_wl, spec_normalized).astype(np.float32)

    try:
        wavelengths, spectrum = _prepare_asc_file(asc_file_path, dataset_path)
    except Exception as e:
        logger.error(f"Gagal memproses file .asc: {e}")
        return

    class_names = list(element_map.keys())
    device = next(model.parameters()).device
    input_tensor = torch.from_numpy(spectrum[np.newaxis, :, np.newaxis]).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    predictions = (probabilities > threshold).astype(int)
    
    output_filename = f"asc_analysis_{os.path.basename(asc_file_path)}.png"
    output_path = os.path.join(results_dir, output_filename)

    create_static_plot(
        wavelengths, spectrum, predictions, probabilities, class_names, output_path,
        ground_truth=None, peak_height=peak_height,
        title=f"Prediksi pada Data Lapangan ({os.path.basename(asc_file_path)})"
    )

# =============================================================================
# BAGIAN 4: PARSER ARGUMEN DAN BLOK EKSEKUSI UTAMA
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Informer Multi-Label Evaluation on HPC")
    parser.add_argument('--model_path', type=str, required=True, help='Path ke file model .pth yang sudah dilatih.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path ke file dataset HDF5.')
    parser.add_argument('--element_map_path', type=str, required=True, help='Path ke file peta elemen JSON.')
    parser.add_argument('--results_dir', type=str, required=True, help='Direktori untuk menyimpan hasil (plot, laporan).')
    
    # Argumen untuk mode evaluasi
    parser.add_argument('--mode', type=str, required=True, choices=['eval', 'analyze_h5', 'analyze_asc'], 
                        help="Mode eksekusi: 'eval' untuk laporan kuantitatif, 'analyze_h5' untuk plot sampel H5, 'analyze_asc' untuk plot data lapangan.")
    
    # Argumen opsional untuk mode analisis
    parser.add_argument('--split', type=str, default='test', help="Split dataset yang digunakan ('train', 'validation', 'test').")
    parser.add_argument('--sample_idx', type=int, default=100, help="Indeks sampel untuk dianalisis (mode analyze_h5).")
    parser.add_argument('--asc_path', type=str, help="Path ke file .asc untuk dianalisis (mode analyze_asc).")
    parser.add_argument('--threshold', type=float, default=0.5, help="Ambang batas untuk klasifikasi.")
    parser.add_argument('--peak_height', type=float, default=0.05, help="Tinggi minimum puncak spektrum untuk dianalisis.")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logging()
    set_seed(42)

    # Buat direktori hasil jika belum ada
    os.makedirs(args.results_dir, exist_ok=True)

    try:
        # Muat model dan peta elemen
        model, element_map = load_analysis_tools(args.model_path, args.element_map_path, CONFIG["model"])

        # Jalankan mode yang dipilih
        if args.mode == 'eval':
            run_final_evaluation(model, element_map, args.dataset_path, args.results_dir, args.split, args.threshold)
        
        elif args.mode == 'analyze_h5':
            analyze_from_h5(model, element_map, args.dataset_path, args.results_dir, 
                            args.sample_idx, args.split, args.peak_height, args.threshold)
            
        elif args.mode == 'analyze_asc':
            if not args.asc_path:
                raise ValueError("Argumen --asc_path diperlukan untuk mode 'analyze_asc'.")
            analyze_from_asc(model, element_map, args.dataset_path, args.results_dir, 
                             args.asc_path, args.peak_height, args.threshold)

    except Exception as e:
        logger.error(f"\n\nPROSES KESELURUHAN GAGAL: {e}")
        traceback.print_exc()
