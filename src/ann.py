import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tkinter import Tk, Button, Label, Entry, Frame, Scrollbar, Listbox, SINGLE, filedialog, StringVar, OptionMenu, \
    RIGHT, Y
from tkinter.ttk import Treeview
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os, sys

if getattr(sys, 'frozen', False):
    # Jika berjalan dari bundle PyInstaller
    bundle_dir = sys._MEIPASS
else:
    # Jika berjalan dalam lingkungan pengembangan
    bundle_dir = os.path.abspath(os.path.dirname(__file__))

db_path = os.path.join(bundle_dir, 'processed_spectra.db')
db_nist = os.path.join(bundle_dir, 'data1.db')
# Fungsi untuk mengambil spektrum eksperimental dalam rentang 200-900 nm
def get_experimental_spectra_200_900(db_path, sample_name):
    conn = sqlite3.connect(db_path)
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
        print(f"Tidak ada data yang ditemukan untuk sampel: {sample_name}")
        return np.array([]), np.array([])

    wavelengths, intensities = zip(*data)
    return np.array(wavelengths), np.array(intensities)


# Fungsi untuk mengambil semua data NIST yang sesuai dengan unsur dan sp_num tertentu
def get_nist_data_for_element_and_sp_num(db_nist, element, sp_num):
    conn = sqlite3.connect(db_nist)
    cursor = conn.cursor()
    query = """
        SELECT "obs_wl_air(nm)", element, sp_num, "gA(s^-1)", "Ek(cm-1)", "Ei(cm-1)", "acc"
        FROM spectrum_data
        WHERE element = ? AND sp_num = ?
    """
    cursor.execute(query, (element, sp_num))
    nist_lines = cursor.fetchall()
    conn.close()

    return nist_lines


# Fungsi untuk melakukan pencocokan data NIST dengan spektrum eksperimen dan mengurutkan berdasarkan intensitas
# Fungsi untuk melakukan pencocokan data NIST dengan spektrum eksperimen dan mengurutkan berdasarkan intensitas
def match_nist_with_experimental(wavelengths, intensities, nist_data, lower, upper):
    matched_data = []
    mask = (wavelengths >= lower) & (wavelengths < upper)
    wl_segment = wavelengths[mask]
    intensity_segment = intensities[mask]

    if len(wl_segment) == 0 or len(intensity_segment) == 0:
        return matched_data

    peaks, _ = find_peaks(intensity_segment)
    peaks_wavelengths = wl_segment[peaks]

    for wl_nist, element, sp_num, gA, Ek, Ei, acc in nist_data:
        # Filter untuk hanya data dengan Accuracy tidak None
        if acc is not None:
            wl_nist = float(wl_nist)
            if lower <= wl_nist < upper:
                match_indices = np.where(np.abs(peaks_wavelengths - wl_nist) < 0.1)[0]
                if len(match_indices) > 0:
                    for idx in match_indices:
                        wl_exp = peaks_wavelengths[idx]
                        intensity_exp = intensity_segment[peaks[idx]]
                        matched_data.append({
                            "Wavelength NIST": wl_nist,
                            "Element": element,
                            "Ion Stage": sp_num,
                            "Experimental Wavelength": wl_exp,
                            "Intensity": intensity_exp,
                            "gA": gA,
                            "Ek": Ek,
                            "Ei": Ei,
                            "Accuracy": acc
                        })

    # Urutkan matched_data berdasarkan 'Intensity' dalam urutan menurun
    matched_data.sort(key=lambda x: x["Intensity"], reverse=True)
    return matched_data


# Fungsi untuk menampilkan hasil pencocokan di dalam tabel
def display_results_in_table(matched_data):
    tree.delete(*tree.get_children())
    for i, row in enumerate(matched_data):
        tree.insert("", "end", iid=i, values=(
            row["Wavelength NIST"], row["Element"], row["Ion Stage"],
            row["Experimental Wavelength"], row["Intensity"],
            row["gA"], row["Ek"], row["Ei"], row["Accuracy"]
        ))


# Fungsi untuk menyimpan hasil yang dipilih ke dalam Excel
# Fungsi untuk menyimpan hasil yang dipilih ke dalam Excel
def save_to_excel():
    selected_items = tree.selection()
    selected_data = []

    for item in selected_items:
        selected_data.append(tree.item(item, "values"))

    if not selected_data:
        print("Tidak ada data yang dipilih untuk disimpan.")
        return

    # Membuat DataFrame dari data yang dipilih
    new_df = pd.DataFrame(selected_data, columns=[
        "Wavelength NIST", "Element", "Ion Stage",
        "Experimental Wavelength", "Intensity",
        "gA", "Ek", "Ei", "Accuracy"
    ])

    # Nama file sesuai dengan sampel yang dipilih
    sample_name = sample_var.get()
    file_path = f"{sample_name}.xlsx"

    try:
        # Jika file sudah ada, kita akan membacanya dan menggabungkan data baru
        existing_df = pd.read_excel(file_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
    except FileNotFoundError:
        # Jika file belum ada, langsung gunakan data baru
        updated_df = new_df

    # Simpan data yang diperbarui ke dalam file Excel yang sama
    updated_df.to_excel(file_path, index=False)
    print("Data telah diperbarui di", file_path)


# Fungsi untuk memplot segmen yang dipilih
def plot_selected_segment(wavelengths, intensities, nist_data, lower, upper):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(lower, upper)
    ax.set_title(f"Spectrum Segment {lower}-{upper} nm")

    mask = (wavelengths >= lower) & (wavelengths < upper)
    wl_segment = wavelengths[mask]
    intensity_segment = intensities[mask]

    if len(wl_segment) == 0 or len(intensity_segment) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12, color='gray')
    else:
        ax.plot(wl_segment, intensity_segment, label='Experimental Data', color='blue')
        peaks, _ = find_peaks(intensity_segment)
        peaks_wavelengths = wl_segment[peaks]

        for wl_nist, element, sp_num, gA, Ek, Ei, acc in nist_data:
            wl_nist = float(wl_nist)
            if lower <= wl_nist < upper:
                match_indices = np.where(np.abs(peaks_wavelengths - wl_nist) < 0.1)[0]
                if len(match_indices) > 0:
                    for idx in match_indices:
                        wl_exp = peaks_wavelengths[idx]
                        intensity_exp = intensity_segment[peaks[idx]]
                        ax.scatter(wl_exp, intensity_exp, color='red')
                        ax.annotate(f"{element} {sp_num}", (wl_exp, intensity_exp), textcoords="offset points",
                                    xytext=(0, 10), ha='center')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.grid(True)
    ax.legend(loc="upper right")

    for widget in plot_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Fungsi untuk menangani pilihan segmen dari Listbox
def on_segment_selected(event=None):
    selection = listbox.curselection()
    if not selection:
        return
    index = selection[0]
    lower, upper = segment_ranges[index]

    element = element_entry.get().strip()
    sp_num = int(ion_stage_var.get())
    sample_name = sample_var.get()

    wavelengths, intensities = get_experimental_spectra_200_900(db_path, sample_name)
    nist_data = get_nist_data_for_element_and_sp_num(db_nist, element, sp_num)
    matched_data = match_nist_with_experimental(wavelengths, intensities, nist_data, lower, upper)
    display_results_in_table(matched_data)
    plot_selected_segment(wavelengths, intensities, nist_data, lower, upper)


# Pengaturan database dan input
db_path = 'processed_spectra.db'
db_nist = 'data1.db'

# GUI Tkinter
root = Tk()
root.title("Spectrum Analyzer")

# Input pilihan sampel
Label(root, text="Pilih Sample:").pack()
sample_var = StringVar(root)
sample_var.set("S1")
sample_menu = OptionMenu(root, sample_var, *[f"S{i}" for i in range(1, 25)])
sample_menu.pack()

# Input elemen dengan Entry
Label(root, text="Masukkan Element:").pack()
element_entry = Entry(root)
element_entry.pack()

# Input pilihan tahap ion
Label(root, text="Pilih Ion Stage:").pack()
ion_stage_var = StringVar(root)
ion_stage_var.set("1")
ion_stage_menu = OptionMenu(root, ion_stage_var, "1", "2", "3", "4", "5")
ion_stage_menu.pack()

# Daftar segmen 50 nm
segment_ranges = [(200 + i * 50, 250 + i * 50) for i in range(14)]
segment_labels = [f"{start}-{end} nm" for start, end in segment_ranges]

# Listbox untuk memilih segmen
Label(root, text="Pilih Segmen (50 nm):").pack()
listbox = Listbox(root, selectmode=SINGLE, exportselection=False, height=6)
for label in segment_labels:
    listbox.insert("end", label)
listbox.bind("<<ListboxSelect>>", on_segment_selected)
listbox.pack(pady=5)

# Frame untuk tabel dan scrollbar
frame = Frame(root)
frame.pack(pady=10)

# Membuat tabel Treeview untuk menampilkan data yang cocok
columns = ["Wavelength NIST", "Element", "Ion Stage", "Experimental Wavelength",
           "Intensity", "gA", "Ek", "Ei", "Accuracy"]

tree = Treeview(frame, columns=columns, show="headings", selectmode="extended")
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=100)

scrollbar = Scrollbar(frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side=RIGHT, fill=Y)
tree.pack()

# Tombol untuk menyimpan data yang dipilih
save_button = Button(root, text="Simpan ke Excel", command=save_to_excel)
save_button.pack(pady=10)

# Frame untuk menampilkan plot di dalam Tkinter
plot_frame = Frame(root)
plot_frame.pack(pady=10)

# Menjalankan GUI
root.mainloop()