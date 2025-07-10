import torch
import torch.nn as nn
import h5py
import numpy as np
import json
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import math

# --- CONFIG (copied from train.py for model definition) ---
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
        "num_classes": 18 # This will be updated from element_map
    }
}

# --- KELAS DATASET (copied from train.py) ---
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
        print(f"Dataset '{self.split}' diinisialisasi dengan {self.dataset_len} sampel.")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
        spectra = self.h5_file[self.split]['spectra'][idx][..., np.newaxis].astype(np.float32)
        labels = self.h5_file[self.split]['labels'][idx].astype(np.float32)
        return torch.from_numpy(spectra), torch.from_numpy(labels)

# --- ARSITEKTUR MODEL (copied from train.py, modified for feature extraction) ---
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

    def extract_features(self, x):
        # This method extracts features from the last encoder layer
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        # Return the output of the last encoder layer, averaged across sequence length
        return x.mean(dim=1) # (batch_size, d_model)

# --- Main Script ---
def set_seed(seed_value=42):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(42)

# Paths
project_root = '/Volumes/Private-C/Private-C-backup/skripsi-21'
model_path = os.path.join(project_root, 'engine', 'informer_multilabel_model-3.pth')
dataset_path = os.path.join(project_root, 'engine', 'data', 'dataset-50.h5') # Corrected dataset path
element_map_path = os.path.join(project_root, 'engine', 'data', 'element-map-18a.json') # Corrected element map path
output_image_path = os.path.join(project_root, 'images', 'tsne_features_plot.png')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load element map to get num_classes and class names
try:
    with open(element_map_path, 'r') as f:
        element_map = json.load(f)
    class_names = list(element_map.keys())
    CONFIG['model']['num_classes'] = len(class_names)
    print(f"Loaded {len(class_names)} classes from element map.")
except FileNotFoundError:
    print(f"Error: element_map.json not found at {element_map_path}. Using default num_classes.")
    class_names = [f'Class {i}' for i in range(CONFIG['model']['num_classes'])] # Fallback

# Initialize model
model = InformerModel(**CONFIG['model']).to(device)

# Load trained weights
if os.path.exists(model_path):
    print(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    # Clean state_dict if it came from torch.compile
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[10:] if k.startswith('_orig_mod.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval() # Set to evaluation mode
    print("Model loaded successfully.")
else:
    print(f"Error: Model weights not found at {model_path}. Cannot perform t-SNE.")
    exit()

# Prepare dataset for feature extraction
# Using validation split for t-SNE visualization
# Limiting samples for t-SNE to make it computationally feasible and visually clear
num_tsne_samples = 2000 # Adjust as needed
try:
    dataset = HDF5SpectrumDataset(file_path=dataset_path, split="validation", max_samples=num_tsne_samples)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
except KeyError:
    print(f"Error: 'validation' split not found in {dataset_path}. Trying 'train' split.")
    try:
        dataset = HDF5SpectrumDataset(file_path=dataset_path, split="train", max_samples=num_tsne_samples)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    except KeyError:
        print(f"Error: Neither 'validation' nor 'train' split found in {dataset_path}. Cannot perform t-SNE.")
        exit()


all_features = []
all_labels = []

print("Extracting features...")
with torch.no_grad():
    for data, labels in tqdm(data_loader, desc="Extracting Features"):
        data = data.to(device)
        features = model.extract_features(data)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

features_np = np.concatenate(all_features, axis=0)
labels_np = np.concatenate(all_labels, axis=0)

print(f"Features shape: {features_np.shape}")
print(f"Labels shape: {labels_np.shape}")

# Perform t-SNE
print("Performing t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(features_np)

print("Plotting t-SNE results...")
plt.figure(figsize=(12, 10))

# Aggregate labels: if any position in sequence has a label, consider the sample to have that label
# labels_np is (num_samples, seq_length, num_classes)
aggregated_labels = (labels_np.sum(axis=1) > 0).astype(int) # (num_samples, num_classes)

num_classes = CONFIG['model']['num_classes']
colors = plt.cm.get_cmap('tab20', num_classes) # Use a colormap for distinct colors

for i in range(num_classes):
    indices = np.where(aggregated_labels[:, i] == 1)[0]
    if len(indices) > 0:
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                    color=colors(i), label=class_names[i], alpha=0.7, s=10)

plt.title('t-SNE Visualization of Informer Model Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid(True, linestyle=':')
plt.tight_layout()

os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
plt.savefig(output_image_path)
print(f"t-SNE plot saved to: {output_image_path}")
plt.close()
