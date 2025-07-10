import h5py
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches
from collections import defaultdict
import matplotlib as mpl
import scipy.signal

# Set Matplotlib untuk tampilan ilmiah
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 9

def generate_plots_and_latex(h5_path, map_path, num_plots=5, output_dir="/Volumes/Data/birrulwldain/spectral_report_v12"):
    """
    Membuka file HDF5, menghasilkan plot spektrum sebagai gambar PNG terpisah,
    dan membuat dokumen LaTeX dengan setiap plot dalam satu halaman penuh horizontal.

    Parameters:
    - h5_path: Path ke file HDF5 dataset
    - map_path: Path ke file JSON element map
    - num_plots: Jumlah plot acak yang akan dibuat
    - output_dir: Direktori untuk menyimpan file output
    """
    # Validasi file input
    if not os.path.exists(h5_path):
        print(f"Error: File dataset tidak ditemukan di {h5_path}")
        return
    if not os.path.exists(map_path):
        print(f"Error: File element map tidak ditemukan di {map_path}")
        return

    # Buat direktori output
    os.makedirs(output_dir, exist_ok=True)
    
    # Muat data pendukung
    with open(map_path, 'r') as f:
        element_map = json.load(f)
    
    idx_to_element = {np.argmax(v): k for k, v in element_map.items()}
    element_to_idx = {k: np.argmax(v) for k, v in element_map.items()}

    # Inisialisasi data untuk LaTeX
    latex_data = []

    with h5py.File(h5_path, 'r') as f:
        if 'train' not in f:
            print("Error: Grup 'train' tidak ditemukan dalam file HDF5.")
            return
            
        train_group = f['train']
        num_train_samples = train_group['spectra'].shape[0]
        
        if num_train_samples == 0:
            print("Peringatan: Tidak ada sampel dalam grup 'train'.")
            return

        if num_train_samples < num_plots:
            print(f"Peringatan: Jumlah sampel ({num_train_samples}) lebih sedikit dari jumlah plot yang diminta ({num_plots}).")
            num_plots = num_train_samples

        random_indices = random.sample(range(num_train_samples), num_plots)
        wavelengths = f['wavelengths'][:]
        
        colors_cmap = plt.colormaps['tab20']
        background_idx = element_to_idx.get('background', -1)
        
        for sample_idx in random_indices:
            spectrum = train_group['spectra'][sample_idx]
            labels = train_group['labels'][sample_idx]
            
            # Dapatkan metadata
            metadata = {}
            sample_id = f"Sample_{sample_idx:04d}"
            temp, n_e = 'N/A', 'N/A'
            initial_composition_data = {}
            try:
                metadata_str = train_group['atom_percentages'][sample_idx].decode('utf-8')
                metadata = json.loads(metadata_str)
                sample_id = metadata.get('sample_id', f"Sample_{sample_idx:04d}")
                temp = metadata.get('temperature', 'N/A')
                n_e = metadata.get('electron_density', 'N/A')
                
                for ion_key, perc in metadata.items():
                    if ion_key not in ['temperature', 'electron_density', 'delta_E_max', 'sample_id']:
                        initial_composition_data[ion_key] = perc

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(f"Peringatan: Gagal memuat metadata untuk ID {sample_idx}: {e}")

            # Hitung posisi aktual (jumlah piksel)
            actual_positions = defaultdict(int)
            total_pixels = labels.shape[0]
            for label_idx in range(labels.shape[1]):
                element_name = idx_to_element.get(label_idx)
                if element_name:
                    active_pixels = np.sum(labels[:, label_idx])
                    actual_positions[element_name] = active_pixels
            
            total_sum_positions = sum(actual_positions.values())

            # Hitung rasio sinyal-ke-noise sederhana (saran informasi tambahan)
            noise_level = np.std(spectrum)
            signal_level = np.max(spectrum)
            snr = signal_level / noise_level if noise_level > 0 else 0

            # --- Generate Plot ---
            fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=300)  # Ukuran lebih besar untuk penuh halaman
            
            ax.plot(wavelengths, spectrum, color='black', linewidth=1.2, label='Spektrum Gabungan')
            legend_handles = [mpatches.Patch(color='black', label='Spektrum Gabungan')]
            
            # --- MODIFIKASI DIMULAI: Anotasi Lebih Baik dengan Pengelompokan ---
            # Struktur untuk menyimpan semua puncak yang terdeteksi, dengan informasi elemen dan intensitas
            all_element_peaks = [] 

            for element_idx in range(labels.shape[1]):
                if element_idx == background_idx:
                    continue

                element_name = idx_to_element.get(element_idx)
                if not element_name:
                    continue

                pixels_with_this_element = np.where(labels[:, element_idx] == 1)[0]
                
                if len(pixels_with_this_element) > 0:
                    spectrum_portion = np.zeros_like(spectrum)
                    spectrum_portion[pixels_with_this_element] = spectrum[pixels_with_this_element]

                    color = colors_cmap(element_idx / len(element_map))
                    ax.fill_between(wavelengths, 0, spectrum_portion, 
                                    facecolor=color, alpha=0.3, label=f'{element_name}')
                    ax.plot(wavelengths, spectrum_portion, color=color, linewidth=0.6)
                    
                    # Identifikasi puncak-puncak penting untuk elemen ini
                    element_spectrum_values = spectrum[pixels_with_this_element]
                    element_wavelengths_for_peak = wavelengths[pixels_with_this_element]

                    # Menggunakan scipy.signal.find_peaks untuk menemukan puncak yang lebih robust
                    # Sesuaikan `height` dan `prominence` sesuai kebutuhan spektrum Anda
                    # `distance` memastikan puncak berjarak minimal 5 piksel
                    # height disesuaikan agar relatif terhadap puncak keseluruhan spektrum
                    peaks, properties = scipy.signal.find_peaks(element_spectrum_values, 
                                                                height=0.05 * np.max(spectrum), 
                                                                prominence=0.01 * np.max(spectrum), 
                                                                distance=5)
                    
                    # Ambil hingga 4 puncak teratas berdasarkan intensitas
                    num_annotations_per_element = 4
                    if len(peaks) > 0:
                        sorted_peak_indices = peaks[np.argsort(element_spectrum_values[peaks])[::-1]] # Urutkan dari intensitas tertinggi

                        annotated_wavelengths_for_element = []
                        for peak_idx_in_element_array in sorted_peak_indices:
                            original_idx_in_spectrum = pixels_with_this_element[peak_idx_in_element_array]
                            current_wavelength = wavelengths[original_idx_in_spectrum]
                            current_intensity = spectrum[original_idx_in_spectrum]

                            # Pastikan tidak ada duplikat anotasi di posisi yang sama atau sangat dekat
                            is_too_close = False
                            for ann_wl in annotated_wavelengths_for_element:
                                if abs(current_wavelength - ann_wl) < 2: # Threshold jarak panjang gelombang (e.g., 2 nm)
                                    is_too_close = True
                                    break
                            
                            if not is_too_close:
                                all_element_peaks.append({
                                    'element': element_name,
                                    'wavelength': current_wavelength,
                                    'intensity': current_intensity,
                                    'color': color
                                })
                                annotated_wavelengths_for_element.append(current_wavelength)
                                if len(annotated_wavelengths_for_element) >= num_annotations_per_element:
                                    break
                    
                    legend_handles.append(mpatches.Patch(color=color, label=element_name))

            # Background
            background_pixels = np.where(labels[:, background_idx] == 1)[0]
            if len(background_pixels) > 0:
                background_portion = np.zeros_like(spectrum)
                background_portion[background_pixels] = spectrum[background_pixels]
                ax.fill_between(wavelengths, 0, background_portion, facecolor='lightgray', alpha=0.2, label='Latar Belakang')
                legend_handles.append(mpatches.Patch(color='lightgray', label='Latar Belakang'))

            # Proses pengelompokan anotasi
            grouped_annotations = []
            
            # Urutkan semua puncak yang terdeteksi berdasarkan panjang gelombang
            all_element_peaks.sort(key=lambda x: x['wavelength'])

            grouping_wavelength_threshold = 0.000000000001 # Ambang batas (dalam nm) untuk mengelompokkan label
            
            i = 0
            while i < len(all_element_peaks):
                current_peak = all_element_peaks[i]
                group = [current_peak]
                
                j = i + 1
                while j < len(all_element_peaks):
                    next_peak = all_element_peaks[j]
                    if abs(next_peak['wavelength'] - current_peak['wavelength']) < grouping_wavelength_threshold:
                        group.append(next_peak)
                        j += 1
                    else:
                        break
                
                # Buat label gabungan
                elements_in_group = sorted(list(set(p['element'] for p in group)))
                avg_wavelength = np.mean([p['wavelength'] for p in group])
                max_intensity_in_group = np.max([p['intensity'] for p in group])
                
                # Format label: "Elemen1 Elemen2 ... PanjangGelombang (nm)"
                combined_label_text = " ".join(elements_in_group) + f" {avg_wavelength:.2f} nm"
                
                # Ambil warna dari elemen pertama dalam grup
                group_color = group[0]['color']

                grouped_annotations.append({
                    'label_text': combined_label_text,
                    'wavelength': avg_wavelength,
                    'intensity': max_intensity_in_group,
                    'color': group_color
                })
                i = j
            
            # Anotasi titik-titik yang telah digabungkan, dengan penyesuaian posisi dasar untuk mengurangi tumpang tindih
            # Ini adalah solusi sederhana, untuk kasus yang sangat padat mungkin perlu algoritma yang lebih canggih
            used_y_positions = defaultdict(list) # Key: range panjang gelombang, Value: daftar offset Y yang digunakan

            for ann in grouped_annotations:
                label_text = ann['label_text']
                wavelength = ann['wavelength']
                intensity = ann['intensity']
                color = ann['color']

                base_y_offset = 20 # Offset dasar dari titik data
                
                # Coba cari slot Y yang tersedia di sekitar panjang gelombang ini
                # Bagi plot menjadi 'bin' panjang gelombang untuk memeriksa tumpang tindih
                # Gunakan bin yang lebih kecil agar lebih sensitif terhadap tumpang tindih vertikal
                wavelength_bin_size = 10 # bin setiap 3 nm
                wavelength_bin = int(wavelength / wavelength_bin_size) * wavelength_bin_size

                current_y_offset = base_y_offset
                # Iterasi untuk menemukan offset Y yang belum terpakai di bin ini
                # Batasi jumlah iterasi untuk mencegah loop tak terbatas pada plot yang sangat padat
                for _ in range(10): # Maksimal coba 10 slot vertikal
                    is_slot_taken = False
                    for existing_offset in used_y_positions[wavelength_bin]:
                        if abs(current_y_offset - existing_offset) < 10: # Cek jika ada offset Y lain yang terlalu dekat (10 piksel)
                            is_slot_taken = True
                            break
                    if is_slot_taken:
                        current_y_offset += 20 # Pindah ke atas
                    else:
                        break
                
                used_y_positions[wavelength_bin].append(current_y_offset) # Catat offset yang dipakai

                ax.annotate(
                    label_text,
                    (wavelength, intensity),
                    textcoords="offset points", xytext=(0, current_y_offset), ha='center',
                    fontsize=9, color='black', # Warna teks label hitam agar konsisten
                    arrowprops=dict(facecolor=color, shrink=0.05, width=0.5, headwidth=4, alpha=0.7),
                    rotation=90 # Tambahkan ini untuk rotasi 90 derajat
                )
            # --- MODIFIKASI SELESAI ---

            # Kustomisasi plot dengan legenda di sudut atas kanan, transparan
            ax.set_title(f"Simulasi Spektrum (ID: {sample_id})", fontsize=14, pad=15, weight='bold')
            ax.set_xlabel("Panjang Gelombang (nm)", fontsize=12, labelpad=10)
            ax.set_ylabel("Intensitas Ternormalisasi", fontsize=12, labelpad=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.7)
            ax.set_xlim(wavelengths.min(), wavelengths.max())
            ax.set_ylim(0, max(1.0, np.max(spectrum) * 1.1))
            
            # Legenda di sudut atas kanan dengan transparansi
            unique_labels_dict = {handle.get_label(): handle for handle in legend_handles}
            ax.legend(handles=list(unique_labels_dict.values()), loc='upper right', fontsize=7, frameon=True, 
                      edgecolor='black', facecolor='white', framealpha=0.7)

            # Simpan plot sebagai PNG terpisah
            plot_filename = f"spectrum_{sample_id}.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.tight_layout()
            plt.savefig(plot_filepath, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot disimpan: {plot_filepath}")

            # Kumpulkan data untuk LaTeX
            table_rows = []
            sorted_elements = sorted(list(set(k.split('_')[0] for k in initial_composition_data.keys()) | set(actual_positions.keys())))
            if 'background' in sorted_elements:
                sorted_elements.remove('background')
                sorted_elements.append('background')

            for elem_name in sorted_elements:
                initial_comp_val = "N/A"
                if f"{elem_name}_1" in initial_composition_data or f"{elem_name}_2" in initial_composition_data:
                    ion1_perc = initial_composition_data.get(f"{elem_name}_1", 0)
                    ion2_perc = initial_composition_data.get(f"{elem_name}_2", 0)
                    initial_comp_val = f"{ion1_perc + ion2_perc:.2f}"
                elif elem_name == 'background':
                    initial_comp_val = 'N/A'
                else:
                    initial_comp_val = "0.00"

                actual_pos_val = actual_positions.get(elem_name, 0)
                table_rows.append({
                    'element': elem_name,
                    'initial_comp': initial_comp_val,
                    'actual_pos': actual_pos_val
                })

            latex_data.append({
                'sample_id': sample_id,
                'temperature': f"{float(temp):.0f}" if temp != 'N/A' else 'N/A',
                'electron_density': f"{float(n_e):.2e}" if n_e != 'N/A' else 'N/A',
                'total_pixels': total_pixels,
                'snr': f"{snr:.2f}",
                'total_actual': total_sum_positions,
                'table_rows': table_rows,
                'plot_filename': plot_filename
            })

    # --- Generate LaTeX Document ---
    latex_content = r"""
\documentclass[a4paper,11pt]{article}
\usepackage{geometry}
\geometry{margin=1cm} % Margin lebih kecil untuk memaksimalkan ruang plot
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{siunitx}
\usepackage{caption}
\usepackage{times}
\usepackage[utf8]{inputenc}
\usepackage{tocloft}
\usepackage{float}

\title{Spectral Simulation Report}
\author{Automated Analysis}
\date{\today}

\begin{document}

\maketitle
\tableofcontents
\clearpage



"""

    for sample in latex_data:
        sample_id = sample['sample_id']
        temp = sample['temperature']
        n_e = sample['electron_density']
        total_pixels = sample['total_pixels']
        snr = sample['snr']
        total_actual = sample['total_actual']
        table_rows = sample['table_rows']
        plot_filename = sample['plot_filename']

        latex_content += f"""
\\section{{Sample: {sample_id}}}
\\begin{{minipage}}[t]{{0.48\\textwidth}}
    \\subsection{{Simulation Parameters}}
    \\centering
    \\begin{{table}}[H]
        \\centering
        \\small
        \\begin{{tabular}}{{ll}}
            \\toprule
            \\textbf{{Parameter}} & \\textbf{{Value}} \\\\
            \\midrule
            Temperature ($T$) & \\SI{{{temp}}}{{\\kelvin}} \\\\
            Electron Density ($n_e$) & \\SI{{{n_e}}}{{\\per\\cubic\\centi\\meter}} \\\\
            Total Pixels & \\num{{{total_pixels}}} \\\\
            Signal-to-Noise Ratio (SNR) & \\num{{{snr}}} \\\\
            \\bottomrule
        \\end{{tabular}}
        \\caption{{Simulation parameters and additional metrics for sample {sample_id}.}}
        \\label{{tab:params_{sample_id}}}
    \\end{{table}}
\\end{{minipage}}\\hfill
\\begin{{minipage}}[t]{{0.48\\textwidth}}
    \\subsection{{Composition Analysis}}
    \\centering
    \\begin{{table}}[H]
        \\centering
        \\small
        \\begin{{tabular}}{{lcc}}
            \\toprule
            \\textbf{{Element}} & \\textbf{{Initial Composition (\\%)}} & \\textbf{{Actual Positions}} \\\\
            \\midrule
"""

        for row in table_rows:
            element = row['element'].replace('_', '\\_')
            initial_comp = row['initial_comp']
            actual_pos = row['actual_pos']
            latex_content += f"            {element} & {initial_comp} & {actual_pos} \\\\\n"

        latex_content += f"""
            \\midrule
            \\textbf{{Total Positions}} & & {total_actual} \\\\
            \\bottomrule
        \\end{{tabular}}
        \\caption{{Composition analysis for sample {sample_id}.}}
        \\label{{tab:comp_{sample_id}}}
    \\end{{table}}
\\end{{minipage}}
\\subsection{{Spectral Plot}}
\\begin{{figure}}[h]
    \\centering
    \\includegraphics[width=\\textwidth,angle=0]{{/Volumes/Data/birrulwldain/spectral_report_v12/{plot_filename}}}
    \\caption{{Spectral simulation plot for sample {sample_id} with multi-label encoding and detailed composition analysis.}}
    \\label{{fig:spectrum_{sample_id}}}
    \\addcontentsline{{toc}}{{subsection}}{{Spectral Plot for Sample {sample_id}}}
\\end{{figure}}
\\clearpage
"""

    latex_content += r"""
\end{document}
"""

    # Simpan file LaTeX
    latex_filepath = os.path.join(output_dir, "spectral_report.tex")
    with open(latex_filepath, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    print(f"Dokumen LaTeX disimpan: {latex_filepath}")

    print(f"\nInstruksi: Untuk menghasilkan PDF, jalankan perintah berikut di direktori {output_dir}:\n"
          f"pdflatex {latex_filepath}\n"
          "Pastikan pdflatex terinstal dan semua file gambar (PNG) ada di direktori yang sama. Jalankan dua kali untuk memperbarui referensi.")

# --- Konfigurasi Path ---
H5_FILE_PATH = "dataset-20.h5"  # Ganti dengan path dataset Anda
MAP_FILE_PATH = "element-map-18a.json"  # Ganti dengan path element map Anda
OUTPUT_PLOTS_DIR = "spectral_report_v12"  # Direktori output

# Jalankan fungsi
generate_plots_and_latex(H5_FILE_PATH, MAP_FILE_PATH, num_plots=10, output_dir=OUTPUT_PLOTS_DIR)