import numpy as np

def minmax_normalize(spectrum):
    return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

def zscore_normalize(spectrum):
    return (spectrum - np.mean(spectrum)) / np.std(spectrum)