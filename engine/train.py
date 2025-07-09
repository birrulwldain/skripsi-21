import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
import json
import os
import random
import argparse
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
import math
from torch.amp import GradScaler, autocast
import psutil
import optuna
import sys

# Mengatur variabel lingkungan untuk mencegah fragmentasi memori di PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- KONFIGURASI DEFAULT ---
CONFIG = {
    "model": {
        "input_dim": 1, 
        "d_model": 32, 
        "nhead": 4, 
        "num_encoder_layers": 3,
        "dim_feedforward": 64, 
        "dropout": 0.4, 
        "seq_length": 4096,
        "attn_factor": 5, 
        "num_classes": 18
    },
    "training": {
        "batch_size": 25, 
        "num_epochs": 10, 
        "learning_rate": 1e-4, 
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "weight_decay": 1e-5, 
        "gradient_accumulation_steps": 4,
        "early_stopping_patience": 6,
        "min_f1_delta": 0.001,
        "progressive_loading": False,       # Aktifkan/non-aktifkan fitur ini
        "initial_samples": 8000,           # Jumlah sampel di epoch pertama
        "sample_increment": 1000           # Tambahan sampel untuk setiap epoch berikutnya
    }
    
}

# --- KELAS DATASET ---
class HDF5SpectrumDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, split, max_samples=None):
        self.file_path = file_path
        self.split = split
        self.h5_file = None
        with h5py.File(self.file_path, 'r') as f:
            if self.split not in f:
                raise KeyError(f"Split '{self.split}' tidak ditemukan di file HDF5: {self.file_path}")
            total_len = len(f[self.split]['spectra'])
            self.dataset_len = min(total_len, max_samples) if max_samples is not None else total_len
        if 'optuna' not in str(sys.argv):
             print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset '{self.split}' diinisialisasi dengan {self.dataset_len} sampel.")
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
        spectra = self.h5_file[self.split]['spectra'][idx][..., np.newaxis].astype(np.float32)
        labels = self.h5_file[self.split]['labels'][idx].astype(np.float32)
        return torch.from_numpy(spectra), torch.from_numpy(labels)

# --- FUNGSI UTILITAS ---
def log_memory_usage(context=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 ** 3)
    vmem = psutil.virtual_memory()
    total_mem = vmem.total / (1024 ** 3)
    available_mem = vmem.available / (1024 ** 3)
    used_percent = vmem.percent
    gpu_mem_log = ""
    if torch.cuda.is_available():
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_mem = torch.cuda.memory_reserved() / (1024 ** 3)
        gpu_mem_log = (f"\n    Memori GPU: Dialokasikan {allocated_mem:.2f} GB | Dicadangkan {reserved_mem:.2f} GB")
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"--- Status Memori ({context}) ---\n"
        f"    Proses Saat Ini: {rss:.2f} GB | "
        f"RAM Sistem: Digunakan {used_percent}% dari {total_mem:.2f} GB | Tersedia: {available_mem:.2f} GB"
        f"{gpu_mem_log}"
    )

def compute_class_weights(dataset, num_classes, batch_size=256, num_workers=2):
    class_pos_counts = np.zeros(num_classes, dtype=np.float64)
    total_pixels_per_class = 0
    temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Menghitung class weights (multi-label)...")
    for _, labels in tqdm(temp_loader, desc="Calculating Weights"):
        class_pos_counts += labels.sum(dim=(0, 1)).cpu().numpy()
        total_pixels_per_class += labels.shape[0] * labels.shape[1]
    weights = total_pixels_per_class / (class_pos_counts + 1e-6)
    weights = np.clip(weights, 1.0, 50.0)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Perhitungan Class weights selesai.")
    return torch.tensor(weights, dtype=torch.float32)

def plot_and_save_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training & Validation Loss'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, linestyle=':')
    ax2.plot(epochs, history['val_f1_macro'], 'go-', label='Validation F1-Score (Macro)')
    ax2.set_title('Validation F1-Score & Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1-Score'); ax2.grid(True, linestyle=':')
    ax2b = ax2.twinx()
    ax2b.plot(epochs, history['val_acc'], 'ms--', label='Validation Accuracy (Subset)', alpha=0.6)
    ax2b.set_ylabel('Accuracy (Subset)', color='m'); ax2b.tick_params(axis='y', labelcolor='m')
    lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history_plot.png")
    plt.savefig(save_path)
    print(f"\nPlot histori training disimpan di: {save_path}")
    plt.close(fig)

def set_seed(seed_value=42):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# --- ARSITEKTUR MODEL ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
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
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.decoder(x)

# --- FUNGSI LOSS ---
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma, self.pos_weight, self.reduction = gamma, pos_weight, reduction
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_loss = ((1 - pt)**self.gamma * bce_loss)
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

# --- FUNGSI TRAINING & VALIDASI ---
def train_one_epoch(model, data_loader, criterion, optimizer, device, grad_accum, scaler, scheduler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(data_loader, desc="Training", leave=False)
    for i, (data, labels) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(device_type=device, dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target) / grad_accum
        scaler.scale(loss).backward()
        if (i + 1) % grad_accum == 0 or (i + 1) == len(data_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        total_loss += loss.item() * grad_accum
        pbar.set_postfix({'Loss': f'{total_loss / (i + 1):.4f}'})
    return total_loss / len(data_loader)

def validate_one_epoch(model, data_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    all_preds_flat, all_targets_flat = [], []
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Validation", leave=False):
            data, target = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type=device, dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)
            val_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).float()
            all_preds_flat.append(preds.view(-1, num_classes).cpu())
            all_targets_flat.append(target.view(-1, num_classes).cpu())
    
    avg_val_loss = val_loss / len(data_loader)
    all_preds = torch.cat(all_preds_flat).numpy()
    all_targets = torch.cat(all_targets_flat).numpy()
    val_acc = accuracy_score(all_targets, all_preds)
    # ### REVISI: Menggunakan F1 Macro sebagai metrik utama ###
    val_f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return avg_val_loss, val_acc, val_f1_macro

# --- FUNGSI PARSE ARGUMEN ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Informer Multi-Label Training on HPC")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path ke file dataset HDF5.')
    parser.add_argument('--element_map_path', type=str, required=True, help='Path ke file peta elemen JSON.')
    parser.add_argument('--model_dir', type=str, required=True, help='Direktori untuk menyimpan file model (.pth).')
    parser.add_argument('--results_dir', type=str, required=True, help='Direktori untuk menyimpan hasil (plot, log, dll.).')
    parser.add_argument('--epochs', type=int, default=CONFIG['training']['num_epochs'], help='Jumlah epoch training.')
    parser.add_argument('--batch_size', type=int, default=CONFIG['training']['batch_size'], help='Ukuran batch.')
    parser.add_argument('--lr', type=float, default=CONFIG['training']['learning_rate'], help='Learning rate.')
    parser.add_argument('--find-batch-size', action='store_true', help='Jalankan mode pencarian batch size optimal dan keluar.')
    parser.add_argument('--optimize-hparams', type=int, nargs='?', const=100, default=None,
                        help='Jalankan mode optimisasi Optuna. Opsional: tentukan jumlah percobaan (default: 100).')
    parser.add_argument('--resume-from', type=str, default=None, 
                    help='Path ke file model .pth untuk melanjutkan training (jika tidak ada progress.json).')
    
    return parser.parse_args()

# --- FUNGSI UNTUK OPTIMISASI & EVALUASI ---

def objective(trial, args):
    """Fungsi yang dijalankan Optuna untuk setiap percobaan (trial)."""
    device = CONFIG['training']['device']
    
    hparams = {
        'd_model': trial.suggest_categorical('d_model', [16, 32, 64]),
        'nhead': trial.suggest_categorical('nhead', [2, 4, 8]),
        'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 4),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 24, 32])
    }
    
    if hparams['d_model'] % hparams['nhead'] != 0:
        raise optuna.exceptions.TrialPruned("d_model harus bisa dibagi oleh nhead.")

    print(f"\n--- Memulai Trial #{trial.number} ---")
    print(f"Parameter: {hparams}")

    train_dataset = HDF5SpectrumDataset(file_path=args.dataset_path, split="train", max_samples=4000)
    val_dataset = HDF5SpectrumDataset(file_path=args.dataset_path, split="validation", max_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size']*2, shuffle=False, num_workers=2)

    model_config = CONFIG['model'].copy()
    model_config.update(hparams)
    model = InformerModel(**model_config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=hparams['learning_rate'])
    criterion = MultiLabelFocalLoss()

    scaler = GradScaler()
    epochs_for_trial = 5 
    for epoch in range(epochs_for_trial):
        train_one_epoch(model, train_loader, criterion, optimizer, device, 1, scaler, None)
        val_loss, _, val_f1_macro = validate_one_epoch(model, val_loader, criterion, device, model_config['num_classes'])
        
        # ### REVISI: Melaporkan hasil dan melakukan pruning di setiap epoch ###
        trial.report(val_f1_macro, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    print(f"--- Trial #{trial.number} Selesai | F1 Macro Score: {val_f1_macro:.4f} ---")
    return val_f1_macro

def run_hparam_optimization(args):
    """Mengelola studi Optuna untuk mencari hyperparameter terbaik."""
    print("\n--- Memulai Mode Optimisasi Hyperparameter dengan Optuna ---")
    
    # ### REVISI: Menggunakan storage untuk melanjutkan studi yang terputus ###
    study_name = "informer-hparam-study"
    storage_url = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.optimize_hparams)
    
    print("\n\n--- Optimisasi Selesai ---")
    print(f"Jumlah percobaan selesai: {len(study.trials)}")
    best_trial = study.best_trial
    print(f"Skor F1 Macro terbaik: {best_trial.value:.4f}")
    print("Parameter terbaik yang ditemukan:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")

def find_optimal_threshold(model, val_loader, device, num_classes):
    """### REVISI: Fungsi baru untuk mencari ambang batas prediksi optimal."""
    print("\n--- Mencari Ambang Batas Prediksi Optimal pada Set Validasi ---")
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Mengevaluasi probabilitas"):
            data, target = data.to(device), labels.to(device)
            with autocast(device_type=device, dtype=torch.float16):
                output = model(data)
                probs = torch.sigmoid(output)
            all_probs.append(probs.view(-1, num_classes).cpu())
            all_targets.append(target.view(-1, num_classes).cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    best_threshold, best_f1 = 0.5, 0.0
    for threshold in tqdm(np.arange(0.1, 0.9, 0.05), desc="Mencoba ambang batas"):
        preds = (all_probs > threshold).astype(int)
        f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
            
    print(f"Ambang batas optimal ditemukan: {best_threshold:.2f} (dengan F1 Macro: {best_f1:.4f})")
    return best_threshold

def run_final_evaluation(model, dataset_path, element_map, device, optimal_threshold):
    """### REVISI: Fungsi baru untuk evaluasi akhir dengan ambang batas optimal."""
    print(f"\n--- EVALUASI FINAL PADA TEST SET (Ambang Batas: {optimal_threshold:.2f}) ---")
    num_classes = len(element_map)
    try:
        test_dataset = HDF5SpectrumDataset(file_path=dataset_path, split="test")
    except KeyError:
        print("Peringatan: Split 'test' tidak ditemukan. Evaluasi akhir dilewati.")
        return

    test_loader = DataLoader(test_dataset, batch_size=CONFIG['training']['batch_size'] * 2, shuffle=False, num_workers=2)
    
    model.eval()
    all_preds_flat, all_targets_flat = [], []
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Mengevaluasi Test Set"):
            data, target = data.to(device), labels.to(device)
            with autocast(device_type=device, dtype=torch.float16):
                output = model(data)
            preds = (torch.sigmoid(output) > optimal_threshold).float()
            all_preds_flat.append(preds.view(-1, num_classes).cpu())
            all_targets_flat.append(target.view(-1, num_classes).cpu())
            
    all_preds = torch.cat(all_preds_flat).numpy()
    all_targets = torch.cat(all_targets_flat).numpy()
    
    print("\n--- Laporan Klasifikasi Final ---")
    report = classification_report(all_targets, all_preds, target_names=list(element_map.keys()), zero_division=0, digits=4)
    print(report)
    
    # Simpan laporan ke file
    report_path = os.path.join(os.path.dirname(CONFIG['training']['model_save_path']), "final_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Evaluasi dengan ambang batas optimal: {optimal_threshold:.2f}\n\n")
        f.write(report)
    print(f"Laporan evaluasi final disimpan di: {report_path}")

# --- FUNGSI UTAMA (MAIN) ---
def main(args):
    set_seed(42)
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # --- Setup path dan konfigurasi dari argumen ---
    CONFIG['training']['model_save_path'] = os.path.join(args.model_dir, 'informer_multilabel_model-3.pth')
    CONFIG['training']['class_weight_path'] = os.path.join(args.model_dir, 'class_weights_multilabel.pth')
    CONFIG['training']['progress_save_path'] = os.path.join(args.model_dir, 'training_progress-3.json')
    CONFIG['training']['num_epochs'] = args.epochs
    CONFIG['training']['batch_size'] = args.batch_size
    CONFIG['training']['learning_rate'] = args.lr

    with open(args.element_map_path, 'r') as f:
        element_map = json.load(f)
    CONFIG['model']['num_classes'] = len(element_map)

    device = CONFIG["training"]["device"]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- MEMULAI JOB ---")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Menggunakan device: {device.upper()}")
    log_memory_usage("Inisialisasi")
    
    # --- Inisialisasi dataset validasi di luar loop karena ukurannya tetap ---
    val_dataset = HDF5SpectrumDataset(file_path=args.dataset_path, split="validation")
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["training"]["batch_size"] * 2, shuffle=False, 
                            num_workers=2, pin_memory=True, persistent_workers=True)
    
    # --- Dapatkan total sampel training yang tersedia untuk batas atas ---
    with h5py.File(args.dataset_path, 'r') as f:
        total_train_samples = len(f['train']['spectra'])

    # --- Inisialisasi Model dan Optimizer ---
    model = InformerModel(**CONFIG["model"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"], weight_decay=CONFIG["training"]["weight_decay"], fused=(device=='cuda'))
    
    # --- Logika untuk melanjutkan training dari checkpoint ---
    start_epoch = 0
    best_val_f1 = 0.0
    histories = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1_macro': []}
    progress_path = CONFIG['training']['progress_save_path']
    model_path = CONFIG['training']['model_save_path']

    if os.path.exists(progress_path):
        print(f"\nFile progres ditemukan. Mencoba melanjutkan training dari: {progress_path}")
        with open(progress_path, 'r') as f:
            progress_data = json.load(f)
        start_epoch = progress_data['metadata'].get('last_saved_epoch', 0)
        best_val_f1 = progress_data['metadata'].get('best_f1_score', 0.0)
        histories = progress_data.get('histories', histories)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Berhasil memuat model dan melanjutkan dari Epoch {start_epoch + 1}")
        else:
            print("Peringatan: File progres ditemukan tapi file model tidak. Memulai dari awal.")
            start_epoch = 0; best_val_f1 = 0.0
    else:
        print(f"\nTidak ada file progres. Memulai training dari awal.")
    # --- REVISI: LOGIKA BARU UNTUK MEMUAT BOBOT SECARA MANUAL ---
    if args.resume_from and os.path.exists(args.resume_from):
        # Cek ini HANYA jika kita TIDAK melanjutkan dari progress.json (start_epoch masih 0)
        if start_epoch == 0:
            print(f"\nMEMUAT BOBOT MANUAL dari: {args.resume_from}")
            state_dict = torch.load(args.resume_from, map_location=device)
            
            # Kode ini akan membersihkan prefix '_orig_mod.' jika ada, aman untuk semua kasus
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[10:] if k.startswith('_orig_mod.') else k
                new_state_dict[name] = v
            
            # Muat state_dict yang bersih. strict=False lebih aman jika ada perbedaan minor.
            model.load_state_dict(new_state_dict, strict=False)
            print("Bobot model berhasil dimuat. Training akan dimulai dengan bobot ini.")
            # Set F1 score lama Anda agar model tidak langsung menimpa simpanan terbaik
            best_val_f1 = 0.52 

    # --- Kompilasi model jika PyTorch versi 2.0+ ---
    if int(torch.__version__.split('.')[0]) >= 2:
        print(f"Mengompilasi model dengan torch.compile()...")
        model = torch.compile(model)

    # --- Logika untuk memuat atau menghitung class weights ---
    class_weight_path = CONFIG["training"]["class_weight_path"]
    if os.path.exists(class_weight_path):
        print(f"Memuat class weights dari: {class_weight_path}")
        class_weights = torch.load(class_weight_path).to(device)
    else:
        # Perlu dataset lengkap untuk menghitung bobot kelas secara akurat
        full_train_dataset = HDF5SpectrumDataset(file_path=args.dataset_path, split="train")
        print(f"Menghitung class weights baru...")
        class_weights = compute_class_weights(full_train_dataset, CONFIG["model"]["num_classes"], num_workers=2).to(device)
        torch.save(class_weights, class_weight_path)
        print(f"Class weights disimpan di: {class_weight_path}")
        del full_train_dataset # Hapus dari memori setelah selesai

    # --- Inisialisasi criterion, scaler, dan variabel early stopping ---
    criterion = MultiLabelFocalLoss(pos_weight=class_weights, gamma=2.0)
    scaler = GradScaler()
    epochs_no_improve = 0
    patience = CONFIG['training']['early_stopping_patience']
    min_delta = CONFIG['training']['min_f1_delta']

    # --- LOOP TRAINING UTAMA ---
    for epoch in range(start_epoch, CONFIG['training']['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['training']['num_epochs']} ---")
        
        # --- BUAT DATASET & DATALOADER DI DALAM LOOP EPOCH UNTUK PROGRESSIVE LOADING ---
        dl_args = {'num_workers': 2, 'pin_memory': True, 'persistent_workers': True}
        
        if CONFIG['training'].get('progressive_loading', False):
            current_max_samples = CONFIG['training']['initial_samples'] + (epoch * CONFIG['training']['sample_increment'])
            current_max_samples = min(current_max_samples, total_train_samples)
            print(f"Mode Progresif: Menggunakan {current_max_samples} dari {total_train_samples} sampel training.")
            train_dataset = HDF5SpectrumDataset(file_path=args.dataset_path, split="train", max_samples=current_max_samples)
        else:
            print("Mode Standar: Menggunakan semua sampel training.")
            train_dataset = HDF5SpectrumDataset(file_path=args.dataset_path, split="train")

        train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True, **dl_args)
        
        # --- Sesuaikan Scheduler dengan DataLoader baru di setiap epoch ---
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["training"]["learning_rate"], 
                                                  epochs=1, steps_per_epoch=len(train_loader), pct_start=0.3)

        # --- Jalankan satu epoch training dan validasi ---
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, CONFIG['training']['gradient_accumulation_steps'], scaler, scheduler)
        val_loss, val_acc, val_f1_macro = validate_one_epoch(model, val_loader, criterion, device, CONFIG['model']['num_classes'])
        
        # --- Catat hasil dan log ---
        histories['train_loss'].append(train_loss)
        histories['val_loss'].append(val_loss)
        histories['val_acc'].append(val_acc)
        histories['val_f1_macro'].append(val_f1_macro)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HASIL EPOCH {epoch+1} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1_macro:.4f}")
        log_memory_usage(f"Akhir Epoch {epoch+1}")
            
        # --- Logika untuk menyimpan model terbaik dan early stopping ---
        if val_f1_macro > best_val_f1 + min_delta:
            best_val_f1 = val_f1_macro
            # Saat menyimpan model yang di-compile, simpan state dict dari model aslinya
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), model_path)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model terbaik disimpan dengan F1 Macro: {best_val_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # --- Simpan progres ---
        progress_data = {
            'metadata': {'last_saved_epoch': epoch + 1, 'best_f1_score': best_val_f1},
            'histories': histories
        }
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=4)

        if epochs_no_improve >= patience:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Early stopping dipicu setelah {patience} epoch tanpa peningkatan.")
            break
    
    # --- SETELAH TRAINING SELESAI ---
    if histories['train_loss']:
        plot_and_save_history(histories, args.results_dir)

    print("\nTraining selesai. Memuat model terbaik untuk evaluasi akhir.")
    
    # Buat instance model baru untuk memuat bobot yang bersih
    final_model = InformerModel(**CONFIG["model"]).to(device)
    if os.path.exists(model_path):
        final_model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Jika model di-compile saat evaluasi juga, lakukan kompilasi
        if int(torch.__version__.split('.')[0]) >= 2:
             final_model = torch.compile(final_model)

        optimal_threshold = find_optimal_threshold(final_model, val_loader, device, CONFIG['model']['num_classes'])
        run_final_evaluation(final_model, args.dataset_path, element_map, device, optimal_threshold)
    else:
        print("Peringatan: File model terbaik tidak ditemukan. Evaluasi akhir dilewati.")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- JOB SELESAI ---")

# --- TITIK MASUK EKSEKUSI ---
if __name__ == "__main__":
    args = parse_arguments()
    
    if args.find_batch_size:
        find_optimal_batch_size(
            model_config=CONFIG["model"],
            training_config=CONFIG["training"],
            dataset_path=args.dataset_path,
            element_map_path=args.element_map_path
        )
    elif args.optimize_hparams is not None:
        run_hparam_optimization(args)
    else:
        main(args)
