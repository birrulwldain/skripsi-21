import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tkinter import Tk, Button, Label, Entry, Frame, StringVar, OptionMenu, DoubleVar, IntVar, filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks


class DataFetcher:
    def __init__(self, db_nist, db_spectrum):
        self.db_nist = db_nist
        self.db_spectrum = db_spectrum

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

    def get_experimental_data(self, sample_name):
        conn = sqlite3.connect(self.db_spectrum)
        cursor = conn.cursor()
        query = """
            SELECT wavelength, intensity
            FROM processed_spectrum
            WHERE sample_name = ? AND wavelength BETWEEN 200 AND 900
            ORDER BY wavelength
        """
        cursor.execute(query, (sample_name,))
        data = cursor.fetchall()
        conn.close()

        if not data:
            print(f"No data found for sample: {sample_name}")
            return np.array([]), np.array([])

        wavelengths, intensities = zip(*data)
        return np.array(wavelengths, dtype=float), np.array(intensities, dtype=float)

    def get_sample_names(self):
        conn = sqlite3.connect(self.db_spectrum)
        cursor = conn.cursor()
        query = "SELECT DISTINCT sample_name FROM processed_spectrum"
        cursor.execute(query)
        samples = [row[0] for row in cursor.fetchall()]
        conn.close()
        return samples


class ExcelDataLoader:
    def __init__(self):
        self.data = None

    def load_data(self, file_path):
        df = pd.read_excel(file_path)
        required_columns = ["Element", "Ion Stage", "NIST WL", "Einstein Coefficient"]
        if all(column in df.columns for column in required_columns):
            self.data = df
            print("Excel data loaded successfully.")
        else:
            print("Excel file is missing required columns.")

    def get_unique_elements(self):
        if self.data is not None:
            unique_elements = self.data[['Element', 'Ion Stage']].drop_duplicates()
            return unique_elements.values.tolist()
        return []

    def get_reference_peaks(self, element, ion_stage):
        if self.data is not None:
            return self.data[(self.data['Element'] == element) & (self.data['Ion Stage'] == ion_stage)][['NIST WL']]
        return pd.DataFrame(columns=['NIST WL'])


class SpectrumSimulator:
    def __init__(self, nist_data, temperature, resolution=24880):
        self.nist_data = nist_data
        self.temperature = temperature
        self.resolution = resolution

    @staticmethod
    def partition_function(energy_levels, degeneracies, T):
        k_B = 8.617333262145e-5
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
                    Ei = Ei / 8065.544
                    gi = float(gi)
                    energy_levels.append(Ei)
                    degeneracies.append(gi)
                except ValueError:
                    continue

        Z = self.partition_function(energy_levels, degeneracies, self.temperature)

        for wl, gA, Ek, Ei, gi, gk, acc in self.nist_data:
            if all(value is not None for value in [wl, gA, Ek, Ei, gi, gk]):
                try:
                    wl = float(wl)
                    gA = float(gA)
                    Ek = float(Ek) / 8065.544
                    gi = float(gi)
                    gk = float(gk)
                    Aki = gA / gk
                    intensity = self.calculate_intensity(self.temperature, Ek, gk, Aki, Z)
                    sigma = 0.1
                    intensities += intensity * self.gaussian_profile(wavelengths, wl, sigma)
                except ValueError:
                    continue

        return wavelengths, intensities


class SpectrumPlotter:
    def __init__(self, main_frame):
        self.plot_frame = Frame(main_frame)
        self.plot_frame.pack(side="top", fill="both", expand=True)
        self.peak_frame = Frame(main_frame)
        self.peak_frame.pack(side="top", fill="both", expand=True)
        self.selected_sample = None

        save_button = Button(self.peak_frame, text="Save Selected to Excel", command=self.save_selected_to_excel)
        save_button.pack(pady=10)

    def plot_spectrum(self, exp_wavelengths, exp_intensities, sim_wavelengths, sim_intensities, ref_peaks):
        scaler = MinMaxScaler(feature_range=(0, 1))
        exp_intensities_normalized = scaler.fit_transform(exp_intensities.reshape(-1, 1)).flatten()
        sim_intensities_normalized = scaler.fit_transform(sim_intensities.reshape(-1, 1)).flatten()

        lower_bound = lower_bound_var.get()
        upper_bound = upper_bound_var.get()
        ref_peaks_filtered = ref_peaks[(ref_peaks['NIST WL'] >= lower_bound) & (ref_peaks['NIST WL'] <= upper_bound)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(exp_wavelengths, exp_intensities_normalized, label='Experimental Spectrum (Normalized)', color='blue')
        ax.plot(sim_wavelengths, sim_intensities_normalized, label='Simulated Spectrum (Normalized)', color='orange',
                linestyle='--')
        ax.plot(ref_peaks_filtered['NIST WL'], [0.5] * len(ref_peaks_filtered), "rx", label="Reference Peaks")
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title("Experimental and Simulated Spectrum with Filtered Reference Peaks")
        ax.grid(True)
        ax.legend()

        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def display_filtered_peaks(self, matching_peaks):
        columns = ["Sim Peak WL", "NIST WL", "Element", "Ion Stage", "Einstein Coefficient", "Acc", "Exp Peak WL",
                   "Exp Intensity"]

        if not hasattr(self, 'tree'):
            self.tree = ttk.Treeview(self.peak_frame, columns=columns, show='headings')
            for col in columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120)
            self.tree.pack()

        for row in self.tree.get_children():
            self.tree.delete(row)

        for peak in matching_peaks:
            self.tree.insert("", "end", values=[peak.get(col, "") for col in columns])

    def save_selected_to_excel(self):
        selected_items = self.tree.selection()
        data = []
        for item in selected_items:
            values = self.tree.item(item, "values")
            data.append(values)

        columns = ["Sim Peak WL", "NIST WL", "Element", "Ion Stage", "Einstein Coefficient", "Acc", "Exp Peak WL",
                   "Exp Intensity"]
        df = pd.DataFrame(data, columns=columns)

        filename = f"{self.selected_sample}.xlsx"
        try:
            exist = pd.read_excel(filename)
            upd = pd.concat([exist, df], ignore_index=True)
        except FileNotFoundError:
            upd = df

        upd.to_excel(filename, index=False)
        print(f"Data saved to {filename}")


root = Tk()
root.title("Filtered Spectrum Analyzer with Adjustable Parameters")

main_frame = Frame(root)
main_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)

input_frame = Frame(root)
input_frame.pack(side="left", padx=10, pady=10)

sample_var = StringVar(root)
element_var = StringVar(root)
ion_stage_var = IntVar(value=1)
temperature_var = DoubleVar(value=11600)
lower_bound_var = DoubleVar(value=200)
upper_bound_var = DoubleVar(value=900)

Label(input_frame, text="Select Sample:").pack()
sample_menu = OptionMenu(input_frame, sample_var, "Select Sample")
sample_menu.pack()

Label(input_frame, text="Select Element and Ion Stage:").pack()
element_menu = OptionMenu(input_frame, element_var, "Select Element")
element_menu.pack()

Label(input_frame, text="Enter Temperature (K):").pack()
temperature_entry = Entry(input_frame, textvariable=temperature_var)
temperature_entry.pack()

Label(input_frame, text="Lower Bound (nm):").pack()
lower_bound_entry = Entry(input_frame, textvariable=lower_bound_var)
lower_bound_entry.pack()

Label(input_frame, text="Upper Bound (nm):").pack()
upper_bound_entry = Entry(input_frame, textvariable=upper_bound_var)
upper_bound_entry.pack()


def simulate_and_display():
    spectrum_plotter.selected_sample = sample_var.get()
    sample_name = sample_var.get()
    element, ion_stage = element_var.get().split()
    ion_stage = int(ion_stage)
    temperature = temperature_var.get()
    lower_bound = lower_bound_var.get()
    upper_bound = upper_bound_var.get()

    nist_data = data_fetcher.get_nist_data(element, ion_stage)
    exp_wavelengths, exp_intensities = data_fetcher.get_experimental_data(sample_name)
    if exp_wavelengths.size == 0 or exp_intensities.size == 0:
        print(f"No data found for sample: {sample_name}")
        return

    simulator = SpectrumSimulator(nist_data, temperature)
    sim_wavelengths, sim_intensities = simulator.simulate()

    ref_peaks = data_loader.get_reference_peaks(element, ion_stage)
    ref_peaks_filtered = ref_peaks[(ref_peaks['NIST WL'] >= lower_bound) & (ref_peaks['NIST WL'] <= upper_bound)]

    exp_peaks_idx, _ = find_peaks(exp_intensities, height=0.0001, distance=10)
    exp_peaks_wavelengths = exp_wavelengths[exp_peaks_idx]
    exp_peaks_intensities = exp_intensities[exp_peaks_idx]

    exp_mask = (exp_wavelengths >= lower_bound) & (exp_wavelengths <= upper_bound)
    sim_mask = (sim_wavelengths >= lower_bound) & (sim_wavelengths <= upper_bound)
    exp_wavelengths_filtered = exp_wavelengths[exp_mask]
    exp_intensities_filtered = exp_intensities[exp_mask]
    sim_wavelengths_filtered = sim_wavelengths[sim_mask]
    sim_intensities_filtered = sim_intensities[sim_mask]

    matching_peaks = []
    tolerance = 0.5
    for ref_wl in ref_peaks_filtered['NIST WL']:
        ref_wl = float(ref_wl)

        nearest_peak_idx = np.argmin(np.abs(exp_peaks_wavelengths - ref_wl))
        nearest_wl = exp_peaks_wavelengths[nearest_peak_idx]
        nearest_intensity = exp_peaks_intensities[nearest_peak_idx]

        nist_entry = next((entry for entry in nist_data if abs(float(entry[0]) - ref_wl) <= tolerance), None)

        if nist_entry and abs(nearest_wl - ref_wl) <= tolerance:
            matching_peaks.append({
                "Sim Peak WL": nearest_wl,
                "NIST WL": ref_wl,
                "Element": element,
                "Ion Stage": ion_stage,
                "Einstein Coefficient": nist_entry[1],
                "Acc": nist_entry[6],
                "Exp Peak WL": nearest_wl,
                "Exp Intensity": nearest_intensity
            })

    spectrum_plotter.plot_spectrum(exp_wavelengths_filtered, exp_intensities_filtered, sim_wavelengths_filtered,
                                   sim_intensities_filtered, ref_peaks_filtered)
    spectrum_plotter.display_filtered_peaks(matching_peaks)


def load_excel_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        data_loader.load_data(file_path)
        update_element_menu()


def update_element_menu():
    elements = data_loader.get_unique_elements()
    element_menu["menu"].delete(0, "end")
    for element, ion_stage in elements:
        label = f"{element} {ion_stage}"
        element_menu["menu"].add_command(label=label, command=lambda el=label: element_var.set(el))


def update_sample_menu():
    samples = data_fetcher.get_sample_names()
    sample_menu["menu"].delete(0, "end")
    for sample in samples:
        sample_menu["menu"].add_command(label=sample, command=lambda s=sample: sample_var.set(s))


analyze_button = Button(input_frame, text="Analyze Segment", command=simulate_and_display)
analyze_button.pack(pady=10)

data_loader = ExcelDataLoader()
data_fetcher = DataFetcher(db_nist='data1.db', db_spectrum='processed_spectra.db')
spectrum_plotter = SpectrumPlotter(main_frame=main_frame)

load_button = Button(input_frame, text="Load Excel Data", command=load_excel_file)
load_button.pack(pady=5)

update_sample_menu()

root.mainloop()