import os
import numpy as np
import pandas as pd

def load_libs_data(data_dir, wavelength_start=200, wavelength_end=900):
    """Muat data LIBS dari file CSV"""
    spectra = []
    labels = []
    for file in os.listdir(data_dir):
        df = pd.read_csv(os.path.join(data_dir, file))
        spectrum = df['intensity'].values
        label = file.split('_')[0]  # Contoh: 'Fe.csv' -> label 'Fe'
        spectra.append(spectrum)
        labels.append(label)
    return np.array(spectra), np.array(labels)