import tkinter as tk
from tkinter import ttk
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from astropy.modeling import models, fitting
import pandas as pd
import os

# DataFetcher class to fetch NIST data
class DataFetcher:
    def __init__(self, db_nist):
        self.db_nist = db_nist

    def get_nist_data(self, element, sp_num):
        conn = sqlite3.connect(self.db_nist)
        cursor = conn.cursor()
        query = """
            SELECT "obs_wl_air(nm)", "gA(s^-1)", "Ek(cm-1)", "Ei(cm-1)", "g_i", "g_k", "acc"
            FROM spectrum_data
            WHERE element = ? AND sp_num = ?
        """
        cursor.execute(query, (element, sp_num))
        data = cursor.fetchall()
        conn.close()
        return data

# SpectrumSimulator class to simulate atomic spectrum
class SpectrumSimulator:
    def __init__(self, nist_data, temperature, resolution=24880):
        self.nist_data = nist_data
        self.temperature = temperature
        self.resolution = resolution

    @staticmethod
    def partition_function(energy_levels, degeneracies, T):
        k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
        Z = np.sum([g * np.exp(-E / (k_B * T)) for g, E in zip(degeneracies, energy_levels)])
        return Z

    @staticmethod
    def calculate_intensity(T, energy, degeneracy, einstein_coeff, Z):
        k_B = 8.617333262145e-5
        return (degeneracy * np.exp(-energy / (k_B * T)) * einstein_coeff) / Z

    @staticmethod
    def gaussian_profile(x, center, sigma):
        return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def simulate(self):
        wavelengths = np.linspace(200, 900, self.resolution)
        intensities = np.zeros_like(wavelengths)

        energy_levels = []
        degeneracies = []

        for wl, gA, Ek, Ei, gi, gk, acc in self.nist_data:
            if all(value is not None for value in [wl, gA, Ek, Ei, gi, gk]):
                try:
                    wl = float(wl)
                    gA = float(gA)
                    Ek, Ei = float(Ek), float(Ei)
                    Ei = Ei / 8065.544  # Convert from cm^-1 to eV
                    gi = float(gi)
                    energy_levels.append(Ei)
                    degeneracies.append(gi)
                except ValueError:
                    continue

        if not energy_levels:
            return None, None

        Z = self.partition_function(energy_levels, degeneracies, self.temperature)

        for wl, gA, Ek, Ei, gi, gk, acc in self.nist_data:
            if all(value is not None for value in [wl, gA, Ek, Ei, gi, gk]):
                try:
                    wl = float(wl)
                    gA = float(gA)
                    Ek = float(Ek) / 8065.544  # Convert from cm^-1 to eV
                    gi = float(gi)
                    gk = float(gk)
                    Aki = gA / gk
                    intensity = self.calculate_intensity(self.temperature, Ek, gk, Aki, Z)
                    sigma = 0.1  # Adjust sigma as needed
                    intensities += intensity * self.gaussian_profile(wavelengths, wl, sigma)
                except ValueError:
                    continue

        return wavelengths, intensities

# Baseline Correction using ALS
def baseline_als(intensities, lam=1e5, p=0.01, niter=7):
    L = len(intensities)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * intensities)
        w = p * (intensities > z) + (1 - p) * (intensities < z)
    return z

# Load Data from Database
def load_data(db_name, sample_name, iteration):
    conn = sqlite3.connect(db_name)
    query = """
        SELECT wavelength, intensity
        FROM spectrum_data
        WHERE sample_name = ? AND iteration = ?
        ORDER BY wavelength
    """
    cursor = conn.cursor()
    cursor.execute(query, (sample_name, iteration))
    results = cursor.fetchall()
    conn.close()
    if results:
        wavelengths, intensities = zip(*results)
        return np.array(wavelengths), np.array(intensities)
    else:
        return None, None

def adjust_fwhm_lorentz(delta):
    """Adjust the FWHM Lorentzian by delta and update the plot."""
    current_value = float(fwhm_lorentz_var.get())
    new_value = max(0.005, current_value + delta)  # Minimum value set to 0.005
    fwhm_lorentz_var.set(f"{new_value:.3f}")
    perform_deconvolution_and_save_peaks()

def adjust_fwhm_gaussian(delta):
    """Adjust the FWHM Gaussian by delta and update the plot."""
    current_value = float(fwhm_gaussian_var.get())
    new_value = max(0.005, current_value + delta)  # Minimum value set to 0.005
    fwhm_gaussian_var.set(f"{new_value:.3f}")
    perform_deconvolution_and_save_peaks()

def add_fwhm_adjust_buttons(frame):
    """Add buttons to adjust FWHM Lorentzian and Gaussian."""
    # Lorentzian FWHM Adjust Buttons
    ttk.Label(frame, text="FWHM Lorentzian:").grid(row=0, column=0, sticky=tk.W)
    fwhm_lorentz_frame = ttk.Frame(frame)
    fwhm_lorentz_frame.grid(row=0, column=1, sticky=tk.W)
    ttk.Button(fwhm_lorentz_frame, text="-", command=lambda: adjust_fwhm_lorentz(-0.01)).pack(side=tk.LEFT)
    ttk.Entry(fwhm_lorentz_frame, textvariable=fwhm_lorentz_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(fwhm_lorentz_frame, text="+", command=lambda: adjust_fwhm_lorentz(0.01)).pack(side=tk.LEFT)

    # Gaussian FWHM Adjust Buttons
    ttk.Label(frame, text="FWHM Gaussian:").grid(row=1, column=0, sticky=tk.W)
    fwhm_gaussian_frame = ttk.Frame(frame)
    fwhm_gaussian_frame.grid(row=1, column=1, sticky=tk.W)
    ttk.Button(fwhm_gaussian_frame, text="-", command=lambda: adjust_fwhm_gaussian(-0.01)).pack(side=tk.LEFT)
    ttk.Entry(fwhm_gaussian_frame, textvariable=fwhm_gaussian_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(fwhm_gaussian_frame, text="+", command=lambda: adjust_fwhm_gaussian(0.01)).pack(side=tk.LEFT)

# Multi-Peak Deconvolution using Astropy
def multi_peak_deconvolution(wavelengths, intensities, peak_indices, fwhm_lorentz, fwhm_gaussian):
    compound_model = None
    individual_models = []

    for i, peak_idx in enumerate(peak_indices):
        peak_center = wavelengths[peak_idx]
        amplitude_est = intensities[peak_idx]
        peak_range = (peak_center - 0.5, peak_center + 0.5)

        # Define Voigt Model for each peak
        voigt = models.Voigt1D(x_0=peak_center, amplitude_L=amplitude_est,
                               fwhm_L=fwhm_lorentz, fwhm_G=fwhm_gaussian)
        voigt.x_0.min = peak_range[0]
        voigt.x_0.max = peak_range[1]
        voigt.fwhm_L.min = 0.005
        voigt.fwhm_L.max = 1.0
        voigt.fwhm_G.min = 0.005
        voigt.fwhm_G.max = 1.0

        individual_models.append(voigt)

        if compound_model is None:
            compound_model = voigt
        else:
            compound_model += voigt

    # Fit the compound model
    fitter = fitting.LevMarLSQFitter()
    with np.errstate(all='ignore'):
        fitted_model = fitter(compound_model, wavelengths, intensities)

    # Collect peak data
    peak_data = {}
    for i, model in enumerate(individual_models):
        x_0 = model.x_0.value
        amplitude = model.amplitude_L.value
        fwhm_L = model.fwhm_L.value
        fwhm_G = model.fwhm_G.value

        # Calculate area under the peak
        peak_wavelengths = np.linspace(x_0 - 0.5, x_0 + 0.5, 500)
        peak_intensities = model(peak_wavelengths)
        area = trapezoid(peak_intensities, peak_wavelengths)

        peak_data[f"Peak {i+1}"] = {
            'x_0': x_0,
            'amplitude': amplitude,
            'fwhm_L': fwhm_L,
            'fwhm_G': fwhm_G,
            'area': area
        }

    fitted_model_total = fitted_model(wavelengths)
    return fitted_model_total, individual_models, peak_data

# Visualize Fitting
def visualize_fitting(wavelengths, intensities, fitted_model_total, individual_models):
    ax.plot(wavelengths, fitted_model_total, label="Total Fitting", color="black", linestyle="--", linewidth=1.2)

    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    for i, model in enumerate(individual_models):
        peak_label = f"Peak {i+1}"
        peak_wavelengths = np.linspace(model.x_0.value - 1.0, model.x_0.value + 1.0, 500)
        peak_intensities = model(peak_wavelengths)
        ax.plot(
            peak_wavelengths,
            peak_intensities,
            label=peak_label,
            color=colors[i % len(colors)],
            alpha=0.8,
            linewidth=1.0,
        )
    ax.legend(loc='upper right')
    canvas.draw()

# Function to update the result table with new data
def update_result_table(df):
    global result_table, status_label
    # Clear the existing table
    for row in result_table.get_children():
        result_table.delete(row)

    # Insert new data into the table
    for _, row in df.iterrows():
        result_table.insert("", "end", values=(row['x_0 (Peak Center)'], row['Area under Peak']))

    status_label.config(text="Data puncak berhasil ditampilkan di tabel.")

# Perform Deconvolution and Save Selected Peaks
def perform_deconvolution_and_save_peaks():
    global wavelengths, corrected_intensities, peaks, mask

    max_peaks = int(max_peaks_var.get())
    fwhm_lorentz = float(fwhm_lorentz_var.get())
    fwhm_gaussian = float(fwhm_gaussian_var.get())
    height_threshold = float(height_threshold_var.get())

    if max_peaks > len(peaks):
        status_label.config(text="Jumlah puncak melebihi jumlah puncak terdeteksi.")
        return

    peak_indices = peaks[:max_peaks]
    try:
        fitted_model_total, individual_models, peak_data = multi_peak_deconvolution(
            wavelengths, corrected_intensities, peak_indices, fwhm_lorentz, fwhm_gaussian
        )
        visualize_fitting(wavelengths, corrected_intensities, fitted_model_total, individual_models)

        # Get selected peaks from the Listbox
        selected_indices = peak_listbox.curselection()
        if not selected_indices:
            status_label.config(text="Tidak ada puncak yang dipilih untuk disimpan.")
            return

        # Prepare data for Excel
        data_to_save = []
        for idx in selected_indices:
            peak_label = f"Peak {idx+1}"
            data = peak_data[peak_label]
            data_to_save.append({
                'Sample': sample_var.get(),
                'Iteration': iteration_var.get(),
                'Peak Number': idx + 1,
                'x_0 (Peak Center)': data['x_0'],
                'Amplitude': data['amplitude'],
                'FWHM_L': data['fwhm_L'],
                'FWHM_G': data['fwhm_G'],
                'Area under Peak': data['area']
            })

        # Convert to DataFrame
        df_new = pd.DataFrame(data_to_save)

        # Check if the Excel file already exists
        excel_file = 'selected_fitting_peaks.xlsx'
        if os.path.exists(excel_file):
            # Read existing data
            df_existing = pd.read_excel(excel_file)
            # Append new data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # Save combined data to Excel
        df_combined.to_excel(excel_file, index=False)
        status_label.config(text="Data puncak hasil fitting telah ditambahkan ke 'selected_fitting_peaks.xlsx'.")

        # Update the result table immediately (added as the last line)
        update_result_table(df_combined)

    except Exception as e:
        status_label.config(text=str(e))

# Update Plot and Peaks List
def update_plot():
    global fig, ax, wavelengths, corrected_intensities, peaks, mask

    lam = float(lambda_var.get())
    iteration = int(iteration_var.get())
    wavelength_min = float(wavelength_min_var.get())
    wavelength_max = float(wavelength_max_var.get())
    height_threshold = float(height_threshold_var.get())
    sample = sample_var.get()

    wavelengths, intensities = load_data(db_processed, sample, iteration)
    if wavelengths is None:
        status_label.config(text="Data tidak ditemukan untuk iterasi ini.")
        return

    mask = (wavelengths >= wavelength_min) & (wavelengths <= wavelength_max)
    wavelengths = wavelengths[mask]
    intensities = intensities[mask]

    if len(wavelengths) == 0:
        status_label.config(text="Tidak ada data dalam rentang panjang gelombang ini.")
        return

    baseline = baseline_als(intensities, lam=lam)  # Removed 'p' parameter here
    corrected_intensities = intensities - baseline

    peaks, _ = find_peaks(corrected_intensities, height=height_threshold, distance=5, prominence=0.01)

    if len(peaks) == 0:
        status_label.config(text="Tidak ada puncak yang terdeteksi. Periksa parameter deteksi.")
        return

    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(wavelengths, intensities, label='Data Asli', color='blue', alpha=0.7, linewidth=0.7)
    ax.plot(wavelengths, baseline, label='Baseline', color='orange', linestyle='--', alpha=0.7, linewidth=0.7)
    ax.plot(wavelengths, corrected_intensities, label='Setelah Baseline Correction', color='green', alpha=0.7, linewidth=0.7)
    ax.plot(wavelengths[peaks], corrected_intensities[peaks], "x", label="Puncak Terdeteksi", color="purple")

    # Fetch NIST data and simulate atomic spectrum
    element = element_var.get().strip()
    sp_num = int(ion_stage_var.get())
    temperature = float(temperature_var.get())

    data_fetcher = DataFetcher(db_nist='data1.db')
    nist_data = data_fetcher.get_nist_data(element, sp_num)

    if nist_data:
        simulator = SpectrumSimulator(nist_data, temperature)
        sim_wavelengths, sim_intensities = simulator.simulate()

        if sim_wavelengths is not None:
            # Limit simulated wavelengths to the range of sample data
            sim_mask = (sim_wavelengths >= wavelength_min) & (sim_wavelengths <= wavelength_max)
            sim_wavelengths = sim_wavelengths[sim_mask]
            sim_intensities = sim_intensities[sim_mask]

            # Normalize simulated intensities with the maximum of sample data
            max_sample_intensity = np.max(corrected_intensities)
            sim_intensities_normalized = sim_intensities / np.max(sim_intensities) * max_sample_intensity

            ax.plot(sim_wavelengths, sim_intensities_normalized, label=f'Simulated Spectrum ({element} {sp_num})', color='cyan', alpha=0.7, linewidth=0.7)
        else:
            status_label.config(text=f"Tidak ada data NIST untuk {element} {sp_num}")
    else:
        status_label.config(text=f"Tidak ada data NIST untuk {element} {sp_num}")

    ax.set_title(f"Proses Spektrum ({sample}, Iterasi {iteration})")
    ax.set_xlabel("Panjang Gelombang (nm)")
    ax.set_ylabel("Intensitas")
    ax.legend(loc='upper right')
    ax.grid(True)

    fig.tight_layout()
    canvas.draw()
    status_label.config(text=f"{len(peaks)} puncak terdeteksi.")

    # Update peak listbox
    peak_listbox.delete(0, tk.END)
    for i, peak_idx in enumerate(peaks):
        peak_wavelength = wavelengths[peak_idx]
        peak_listbox.insert(tk.END, f"Peak {i+1}: {peak_wavelength:.4f} nm")

# Function to copy selected peaks from the result table
def copy_selected_peaks():
    global result_table, status_label
    selected_items = result_table.selection()
    if not selected_items:
        status_label.config(text="Tidak ada baris yang dipilih untuk disalin.")
        return

    # Format data sebagai tabel dengan pemisah tab dan baris
    copied_text = ""
    for item in selected_items:
        values = result_table.item(item, 'values')
        copied_text += f"{values[0]}\t{values[1]}\n"

    # Salin ke clipboard
    root.clipboard_clear()
    root.clipboard_append(copied_text)
    root.update()  # Pastikan clipboard diperbarui
    status_label.config(text="Data puncak yang dipilih telah disalin ke clipboard.")

def main_app():
    global db_processed, ax, canvas, lambda_var, iteration_var, wavelength_min_var, wavelength_max_var
    global status_label, sample_var, fig
    global max_peaks_var, fwhm_lorentz_var, fwhm_gaussian_var, height_threshold_var
    global peak_listbox, result_table, root
    global element_var, ion_stage_var, temperature_var

    db_processed = 'tanah_vulkanik.db'

    root = tk.Tk()
    root.title("Interaktif Baseline Correction dan Voigt Fitting")

    # Menggunakan PanedWindow untuk membagi area antara kontrol dan plot
    paned_window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
    paned_window.pack(fill=tk.BOTH, expand=True)

    # Frame untuk kontrol
    control_frame = ttk.Frame(paned_window, width=300)
    paned_window.add(control_frame, weight=1)

    # Frame untuk plot
    plot_frame = ttk.Frame(paned_window)
    paned_window.add(plot_frame, weight=4)

    # Menambahkan canvas scroll pada control_frame
    canvas_control = tk.Canvas(control_frame)
    scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas_control.yview)
    scrollable_frame = ttk.Frame(canvas_control)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas_control.configure(
            scrollregion=canvas_control.bbox("all")
        )
    )

    canvas_control.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas_control.configure(yscrollcommand=scrollbar.set)

    canvas_control.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    status_label = ttk.Label(scrollable_frame, text="Masukkan parameter dan tekan 'Update Plot'.")
    status_label.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W)

    ttk.Label(scrollable_frame, text="Sample:").grid(row=1, column=0, sticky=tk.W)
    sample_var = tk.StringVar(value="S1")
    sample_dropdown = ttk.Combobox(scrollable_frame, textvariable=sample_var, state="readonly")
    sample_dropdown['values'] = [f"S{i}" for i in range(1, 25)]
    sample_dropdown.grid(row=1, column=1, sticky=tk.W, pady=2)

    ttk.Label(scrollable_frame, text="Lambda (λ):").grid(row=2, column=0, sticky=tk.W)
    lambda_var = tk.StringVar(value="1e5")
    ttk.Entry(scrollable_frame, textvariable=lambda_var).grid(row=2, column=1, sticky=tk.W, pady=2)

    ttk.Label(scrollable_frame, text="Iteration:").grid(row=3, column=0, sticky=tk.W)
    iteration_var = tk.StringVar(value="1")
    iteration_dropdown = ttk.Combobox(scrollable_frame, textvariable=iteration_var, state="readonly")
    iteration_dropdown['values'] = [1, 2, 3]
    iteration_dropdown.grid(row=3, column=1, sticky=tk.W, pady=2)

    ttk.Label(scrollable_frame, text="Wavelength Min:").grid(row=4, column=0, sticky=tk.W)
    wavelength_min_var = tk.StringVar(value="290")
    ttk.Entry(scrollable_frame, textvariable=wavelength_min_var).grid(row=4, column=1, sticky=tk.W, pady=2)

    ttk.Label(scrollable_frame, text="Wavelength Max:").grid(row=5, column=0, sticky=tk.W)
    wavelength_max_var = tk.StringVar(value="300")
    ttk.Entry(scrollable_frame, textvariable=wavelength_max_var).grid(row=5, column=1, sticky=tk.W, pady=2)

    ttk.Label(scrollable_frame, text="Jumlah Puncak Maks:").grid(row=6, column=0, sticky=tk.W)
    max_peaks_var = tk.StringVar(value="5")
    ttk.Entry(scrollable_frame, textvariable=max_peaks_var).grid(row=6, column=1, sticky=tk.W, pady=2)

    ttk.Label(scrollable_frame, text="FWHM Lorentzian:").grid(row=7, column=0, sticky=tk.W)
    fwhm_lorentz_var = tk.StringVar(value="0.05")
    fwhm_lorentz_frame = ttk.Frame(scrollable_frame)
    fwhm_lorentz_frame.grid(row=7, column=1, sticky=tk.W, pady=2)
    ttk.Button(fwhm_lorentz_frame, text="-", command=lambda: adjust_fwhm_lorentz(-0.01)).pack(side=tk.LEFT)
    ttk.Entry(fwhm_lorentz_frame, textvariable=fwhm_lorentz_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(fwhm_lorentz_frame, text="+", command=lambda: adjust_fwhm_lorentz(0.01)).pack(side=tk.LEFT)

    ttk.Label(scrollable_frame, text="FWHM Gaussian:").grid(row=8, column=0, sticky=tk.W)
    fwhm_gaussian_var = tk.StringVar(value="0.05")
    fwhm_gaussian_frame = ttk.Frame(scrollable_frame)
    fwhm_gaussian_frame.grid(row=8, column=1, sticky=tk.W, pady=2)
    ttk.Button(fwhm_gaussian_frame, text="-", command=lambda: adjust_fwhm_gaussian(-0.01)).pack(side=tk.LEFT)
    ttk.Entry(fwhm_gaussian_frame, textvariable=fwhm_gaussian_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(fwhm_gaussian_frame, text="+", command=lambda: adjust_fwhm_gaussian(0.01)).pack(side=tk.LEFT)

    ttk.Label(scrollable_frame, text="Peak Height Threshold:").grid(row=9, column=0, sticky=tk.W)
    height_threshold_var = tk.StringVar(value="0.05")
    ttk.Entry(scrollable_frame, textvariable=height_threshold_var).grid(row=9, column=1, sticky=tk.W, pady=2)

    # Add Element and Ion Stage selection
    ttk.Label(scrollable_frame, text="Element:").grid(row=10, column=0, sticky=tk.W)
    element_var = tk.StringVar(value="Fe")
    ttk.Entry(scrollable_frame, textvariable=element_var).grid(row=10, column=1, sticky=tk.W, pady=2)

    ttk.Label(scrollable_frame, text="Ion Stage:").grid(row=11, column=0, sticky=tk.W)
    ion_stage_var = tk.StringVar(value="1")
    ion_stage_dropdown = ttk.Combobox(scrollable_frame, textvariable=ion_stage_var, state="readonly")
    ion_stage_dropdown['values'] = [str(i) for i in range(1, 6)]
    ion_stage_dropdown.grid(row=11, column=1, sticky=tk.W, pady=2)


    ttk.Label(scrollable_frame, text="Temperature (K):").grid(row=12, column=0, sticky=tk.W)
    temperature_var = tk.StringVar(value="11600")
    ttk.Entry(scrollable_frame, textvariable=temperature_var).grid(row=12, column=1, sticky=tk.W, pady=2)

    update_button = ttk.Button(scrollable_frame, text="Update Plot", command=update_plot)
    update_button.grid(row=13, column=0, columnspan=2, pady=5, sticky=tk.W)

    # Listbox for peak selection
    ttk.Label(scrollable_frame, text="Pilih Puncak untuk Disimpan:").grid(row=14, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
    peak_listbox = tk.Listbox(scrollable_frame, selectmode=tk.MULTIPLE, height=10)
    peak_listbox.grid(row=15, column=0, columnspan=2, sticky=tk.W+tk.E)

    deconvolution_button = ttk.Button(scrollable_frame, text="Perform Deconvolution and Save Peaks",
                                      command=perform_deconvolution_and_save_peaks)
    deconvolution_button.grid(row=16, column=0, columnspan=2, pady=5, sticky=tk.W)

    # Tabel untuk menampilkan hasil
    ttk.Label(scrollable_frame, text="Hasil Puncak yang Disimpan:").grid(row=17, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
    result_table = ttk.Treeview(scrollable_frame, columns=("Peak Center", "Area"), show="headings", height=10)
    result_table.heading("Peak Center", text="Peak Center (x_0)")
    result_table.heading("Area", text="Area under Peak")
    result_table.column("Peak Center", anchor=tk.CENTER, width=120)
    result_table.column("Area", anchor=tk.CENTER, width=120)
    result_table.grid(row=18, column=0, columnspan=2, sticky=tk.W+tk.E)

    # Tombol untuk menyalin hasil
    copy_button = ttk.Button(scrollable_frame, text="Copy Selected Peaks", command=copy_selected_peaks)
    copy_button.grid(row=19, column=0, columnspan=2, pady=5, sticky=tk.W)

    # Plot area
    fig = plt.figure(figsize=(10, 5))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    root.mainloop()

if __name__ == "__main__":
    main_app()