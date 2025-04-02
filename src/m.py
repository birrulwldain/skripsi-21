import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tkinter import Tk, Button, Label, filedialog, StringVar, OptionMenu, messagebox, ttk, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks

k_B = 8.617333262145e-5


def get_ek_from_nist(db_nist, nist_wavelength):
    conn = sqlite3.connect(db_nist)
    cursor = conn.cursor()
    query = """
        SELECT "Ek(cm-1)"
        FROM spectrum_data
        WHERE "obs_wl_air(nm)" = ?
    """
    cursor.execute(query, (nist_wavelength,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return float(result[0]) / 8065.544
    else:
        return None


def load_data_from_excel(file_path, db_nist, selected_element, selected_ion_stage):
    df = pd.read_excel(file_path)
    required_columns = ["Element", "Ion Stage", "NIST WL", "Exp Peak WL"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Excel file is missing required columns.")

    filtered_data = df[(df["Element"] == selected_element) & (df["Ion Stage"] == selected_ion_stage)]
    nist_wavelengths = filtered_data["NIST WL"].tolist()
    exp_wavelengths = filtered_data["Exp Peak WL"].tolist()
    ek_values = [get_ek_from_nist(db_nist, wl) for wl in nist_wavelengths]

    data = []
    for nist_wl, exp_wl, ek in zip(nist_wavelengths, exp_wavelengths, ek_values):
        ek_display = f"{ek:.4f} eV" if ek is not None else "N/A"
        data.append((nist_wl, exp_wl, ek_display))
    return data


def get_nist_data_for_wavelengths(db_nist, nist_wavelengths):
    conn = sqlite3.connect(db_nist)
    cursor = conn.cursor()
    nist_data = []
    for wl in nist_wavelengths:
        query = """
            SELECT "obs_wl_air(nm)", "Ek(cm-1)", "gA(s^-1)"
            FROM spectrum_data
            WHERE "obs_wl_air(nm)" = ?
        """
        cursor.execute(query, (wl,))
        result = cursor.fetchone()
        if result:
            obs_wl, Ek, gA = result
            try:
                Ek = float(Ek) / 8065.544  # Konversi ke eV
                nist_data.append((obs_wl, Ek, float(gA)))
            except ValueError:
                print(f"Skipping invalid Ek value for wavelength {obs_wl}")
    conn.close()
    return nist_data


def get_integrated_intensity_with_min_bounds(db_processed, sample_name, exp_wavelengths):
    conn = sqlite3.connect(db_processed)
    cursor = conn.cursor()
    integrated_intensities = []

    for wl_exp in exp_wavelengths:
        query = """
            SELECT wavelength, intensity 
            FROM processed_spectrum
            WHERE sample_name = ? AND wavelength BETWEEN ? AND ?
            ORDER BY wavelength
        """
        width = 0.2
        cursor.execute(query, (sample_name, wl_exp - width, wl_exp + width))
        results = cursor.fetchall()

        if results:
            wavelengths, intensities = zip(*results)
            peak_index, _ = find_peaks(intensities, height=max(intensities) * 0.5)
            if len(peak_index) == 0:
                integrated_intensities.append(None)
                continue

            peak_index = peak_index[0]
            left_min_index = np.argmin(intensities[:peak_index])
            right_min_index = np.argmin(intensities[peak_index:]) + peak_index

            integrated_area = np.trapz(intensities[left_min_index:right_min_index + 1],
                                       wavelengths[left_min_index:right_min_index + 1])
            integrated_intensities.append(integrated_area)
        else:
            integrated_intensities.append(None)
    conn.close()
    return integrated_intensities


def calculate_temperature(nist_data, intensities, exp_wavelengths):
    energies = []
    boltzmann_values = []

    for i, (wl_nist, Ek, gA) in enumerate(nist_data):
        intensity = intensities[i]
        if intensity is None or gA == 0:
            continue

        boltzmann_value = np.log((intensity * exp_wavelengths[i]) / gA)
        energies.append(Ek)
        boltzmann_values.append(boltzmann_value)

    if len(boltzmann_values) < 2:
        print("Tidak cukup data untuk menghitung suhu plasma.")
        return None, None, None

    slope, intercept, _, _, _ = linregress(energies, boltzmann_values)
    T_plasma = -1 / (k_B * slope)
    return T_plasma, slope, energies, boltzmann_values


class PlasmaTemperatureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plasma Temperature Calculator")
        self.root.geometry("1000x600")  # Initial window size

        # Configure root grid to make it responsive
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.db_nist = "data1.db"
        self.db_processed = "processed_spectra.db"
        self.excel_file_path = None
        self.excel_data = None
        self.elements = []
        self.ion_stages = {}
        self.selected_element = StringVar()
        self.selected_ion_stage = StringVar()
        self.selected_sample = StringVar()
        self.selected_lines = []

        main_frame = Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Make main_frame expandable
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=3)

        self.setup_ui(main_frame)

    def setup_ui(self, frame):
        # Set up the input frame
        input_frame = Frame(frame)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ns")
        input_frame.grid_rowconfigure(6, weight=1)  # Allows resizing vertically

        load_button = Button(input_frame, text="Load Excel File", command=self.load_excel_file)
        load_button.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")

        Label(input_frame, text="Select Element:").grid(row=1, column=0, sticky="e")
        self.element_menu = OptionMenu(input_frame, self.selected_element, ())
        self.element_menu.grid(row=1, column=1, sticky="ew", pady=5)

        Label(input_frame, text="Select Ion Stage:").grid(row=2, column=0, sticky="e")
        self.ion_stage_menu = OptionMenu(input_frame, self.selected_ion_stage, ())
        self.ion_stage_menu.grid(row=2, column=1, sticky="ew", pady=5)

        Label(input_frame, text="Select Sample:").grid(row=3, column=0, sticky="e")
        sample_options = [f"S{i}" for i in range(1, 25)]
        self.selected_sample.set(sample_options[0])
        sample_menu = OptionMenu(input_frame, self.selected_sample, *sample_options)
        sample_menu.grid(row=3, column=1, sticky="ew", pady=5)

        show_table_button = Button(input_frame, text="Show Lines for Selection", command=self.show_line_table)
        show_table_button.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        calculate_button = Button(input_frame, text="Calculate Plasma Temperature",
                                  command=self.calculate_plasma_temperature)
        calculate_button.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

        # Set up the plot frame and make it proportional
        plot_frame = Frame(frame)
        plot_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame = plot_frame
        self.canvas = None

        # Set up the table frame and make it proportional
        self.table_frame = Frame(frame)
        self.table_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.table_frame.grid_rowconfigure(0, weight=1)
        self.table_frame.grid_columnconfigure(0, weight=1)

        # Set up the result frame
        self.result_frame = Frame(frame)
        self.result_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        self.result_frame.grid_rowconfigure(0, weight=1)
        self.result_frame.grid_columnconfigure(0, weight=1)

    def load_excel_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            try:
                self.excel_file_path = file_path
                df = pd.read_excel(file_path)
                required_columns = ["Element", "Ion Stage", "NIST WL", "Exp Peak WL"]
                if not all(col in df.columns for col in required_columns):
                    raise ValueError("Excel file is missing required columns.")
                self.excel_data = df
                self.update_element_menu()
                messagebox.showinfo("Success", "Excel data loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def update_element_menu(self):
        self.elements = self.excel_data["Element"].unique().tolist()
        self.ion_stages = {
            element: self.excel_data[self.excel_data["Element"] == element]["Ion Stage"].unique().tolist()
            for element in self.elements
        }

        self.element_menu["menu"].delete(0, "end")
        for element in self.elements:
            self.element_menu["menu"].add_command(label=element,
                                                  command=lambda el=element: self.on_element_selected(el))

        if self.elements:
            self.selected_element.set(self.elements[0])
            self.on_element_selected(self.elements[0])

    def on_element_selected(self, element):
        self.selected_element.set(element)
        self.update_ion_stage_menu(element)

    def update_ion_stage_menu(self, element):
        ion_stages = self.ion_stages.get(element, [])
        self.ion_stage_menu["menu"].delete(0, "end")

        for ion_stage in ion_stages:
            self.ion_stage_menu["menu"].add_command(label=ion_stage,
                                                    command=lambda ion=ion_stage: self.selected_ion_stage.set(ion))

        if ion_stages:
            self.selected_ion_stage.set(ion_stages[0])

    def show_line_table(self):
        selected_element = self.selected_element.get()
        selected_ion_stage = int(self.selected_ion_stage.get())

        line_data = load_data_from_excel(self.excel_file_path, self.db_nist, selected_element, selected_ion_stage)
        exp_wavelengths = [row[1] for row in line_data]
        integrated_intensities = get_integrated_intensity_with_min_bounds(self.db_processed, self.selected_sample.get(),
                                                                          exp_wavelengths)

        for widget in self.table_frame.winfo_children():
            widget.destroy()

        tree = ttk.Treeview(self.table_frame, columns=("NIST WL", "Exp WL", "Ek (eV)", "Integrated Intensity"),
                            show="headings", selectmode="extended")
        tree.heading("NIST WL", text="NIST WL")
        tree.heading("Exp WL", text="Exp WL")
        tree.heading("Ek (eV)", text="Ek (eV)")
        tree.heading("Integrated Intensity", text="Integrated Intensity")

        for row, intensity in zip(line_data, integrated_intensities):
            tree.insert("", "end", values=(row[0], row[1], row[2], f"{intensity:.2f}" if intensity else "N/A"))

        tree.pack(fill="both", expand=True)

        select_button = Button(self.table_frame, text="Select Lines", command=lambda: self.select_lines(tree))
        select_button.pack(pady=5)

    def select_lines(self, tree):
        selected_items = tree.selection()
        self.selected_lines = [(tree.item(item)["values"][0], tree.item(item)["values"][1]) for item in selected_items]
        messagebox.showinfo("Selection", f"{len(self.selected_lines)} lines selected for temperature calculation.")

    def calculate_plasma_temperature(self):
        if not self.selected_lines:
            messagebox.showerror("Error", "No lines selected for calculation.")
            return

        nist_wavelengths = [float(line[0]) for line in self.selected_lines]
        exp_wavelengths = [float(line[1]) for line in self.selected_lines]

        intensities = get_integrated_intensity_with_min_bounds(self.db_processed, self.selected_sample.get(),
                                                               exp_wavelengths)
        nist_data = get_nist_data_for_wavelengths(self.db_nist, nist_wavelengths)

        T_plasma, slope, energies, boltzmann_values = calculate_temperature(nist_data, intensities, exp_wavelengths)

        if T_plasma:
            messagebox.showinfo("Result", f"Suhu plasma: {T_plasma:.2f} K")
            self.plot_boltzmann_plot(energies, boltzmann_values, slope)
            self.display_results(T_plasma, slope)
            self.export_results(T_plasma, slope, nist_wavelengths, exp_wavelengths, intensities)

    def plot_boltzmann_plot(self, energies, boltzmann_values, slope):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Set transparansi background
        fig.patch.set_alpha(0)  # Background figure transparan
        ax.set_facecolor((1, 1, 1, 0))  # Background axis transparan
        ax.grid(False)  # Menghapus grid

        # Plot data points and fit line
        intercept = boltzmann_values[0] - slope * energies[0]
        ax.scatter(energies, boltzmann_values, color='red', s=10, label="Data Points")  # Titik berwarna merah
        ax.plot(energies, slope * np.array(energies) + intercept, color='black', linewidth=0.4,
                label=f'Fit Line (Slope = {slope:.3f})')  # Garis plot hitam dengan ketebalan 0.4

        # Set warna dan ketebalan border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        ax.set_xlabel('Excitation Energy (eV)')
        ax.set_ylabel('ln(I * λ / gA)')
        ax.set_title('Boltzmann Plot')
        ax.legend()

        # Render plot to Tkinter canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Save plot as PDF with transparency
        fig.savefig(f"{self.selected_sample.get()}_BoltzmannPlot.pdf", format="pdf", transparent=True)

    def display_results(self, T_plasma, slope):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        result_table = ttk.Treeview(self.result_frame, columns=("Slope", "Temperature (K)"), show="headings")
        result_table.heading("Slope", text="Slope")
        result_table.heading("Temperature (K)", text="Temperature (K)")
        result_table.insert("", "end", values=(f"{slope:.3f}", f"{T_plasma:.2f}"))
        result_table.pack(fill="both", expand=True)

    def export_results(self, T_plasma, slope, nist_wavelengths, exp_wavelengths, intensities):
        # Mendapatkan data tambahan untuk ekspor
        nist_data = get_nist_data_for_wavelengths(self.db_nist, nist_wavelengths)
        energies = []
        boltzmann_values = []

        for i, (wl_nist, Ek, gA) in enumerate(nist_data):
            intensity = intensities[i]
            if intensity is not None and gA != 0:
                boltzmann_value = np.log((intensity * exp_wavelengths[i]) / gA)
                energies.append(Ek)
                boltzmann_values.append(boltzmann_value)
            else:
                energies.append(None)
                boltzmann_values.append(None)

        # Menyiapkan data untuk diekspor dalam bentuk DataFrame
        data = {
            "Element": [self.selected_element.get()] * len(nist_wavelengths),
            "Ion Stage": [self.selected_ion_stage.get()] * len(nist_wavelengths),
            "NIST WL": nist_wavelengths,
            "Exp WL": exp_wavelengths,
            "Integrated Intensity": [f"{i:.2f}" if i else "N/A" for i in intensities],
            "Ek (eV)": [
                f"{get_ek_from_nist(self.db_nist, wl):.4f}" if get_ek_from_nist(self.db_nist, wl) is not None else "N/A"
                for wl in nist_wavelengths],
            "Einstein Coefficient": [nist_data[i][2] if i < len(nist_data) else "N/A" for i in
                                     range(len(nist_wavelengths))],
            "Excitation Energy (eV)": energies,
            "Boltzmann Value": boltzmann_values
        }

        # Mengonversi data menjadi DataFrame
        df = pd.DataFrame(data)

        # Menambahkan baris terakhir dengan hasil slope dan suhu sebagai DataFrame tambahan
        results_data = {
            "Element": ["Result"],
            "Ion Stage": ["-"],
            "NIST WL": ["-"],
            "Exp WL": ["-"],
            "Integrated Intensity": ["-"],
            "Ek (eV)": ["-"],
            "Einstein Coefficient": ["-"],
            "Excitation Energy (eV)": ["Slope"],
            "Boltzmann Value": [slope]
        }
        results_df = pd.DataFrame(results_data)
        results_df = pd.concat(
            [results_df, pd.DataFrame({"Excitation Energy (eV)": ["Temperature (K)"], "Boltzmann Value": [T_plasma]})],
            ignore_index=True)

        # Menggabungkan hasil dengan data utama
        df = pd.concat([df, results_df], ignore_index=True)

        # Ekspor DataFrame ke file Excel
        output_filename = f"{self.selected_sample.get()}_T.xlsx"
        df.to_excel(output_filename, index=False)
        messagebox.showinfo("Export", f"Results exported successfully as {output_filename}.")


root = Tk()
app = PlasmaTemperatureApp(root)
root.mainloop()