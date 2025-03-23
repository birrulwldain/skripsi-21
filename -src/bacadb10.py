import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import least_squares
import time
from datetime import datetime
import pytz


class baca:

    @staticmethod
    def nist(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                "obs_wl_air(nm)", 
                "element", 
                "sp_num", 
                "acc"
            FROM spectrum_data
            WHERE "sp_num" IN (1, 2)
            """
        )
        data = cursor.fetchall()
        conn.close()

        if not data:
            print("Tidak ada data ditemukan di database.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Filter out rows where 'acc' is NaN or empty
        filtered_data = [row for row in data if row[3] is not None and row[3] != '']

        if not filtered_data:
            print("Tidak ada data yang memenuhi syarat.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        nist_wavelengths, nist_elements, nist_nums, nist_acc = zip(*filtered_data)
        nist_wavelengths = np.array([float(w) if w else np.nan for w in nist_wavelengths])
        nist_elements = np.array(nist_elements)
        nist_nums = np.array(nist_nums)
        nist_acc = np.array([str(a) for a in nist_acc])

        return nist_wavelengths, nist_elements, nist_nums, nist_acc

    @staticmethod
    def spec(db_path, sample_name, lower_bound=None, upper_bound=None):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = """
            SELECT wavelength, AVG(intensity) as avg_intensity
            FROM spectrum_data
            WHERE sample_name = ?
        """
        params = [sample_name]
        if lower_bound is not None and upper_bound is not None:
            query += " AND wavelength BETWEEN ? AND ?"
            params.extend([lower_bound, upper_bound])
        elif lower_bound is not None:
            query += " AND wavelength >= ?"
            params.append(lower_bound)
        elif upper_bound is not None:
            query += " AND wavelength <= ?"
            params.append(upper_bound)
        query += " GROUP BY wavelength ORDER BY wavelength"
        cursor.execute(query, params)
        data = cursor.fetchall()
        conn.close()
        if not data:
            print(f"Tidak ada data ditemukan untuk sampel: {sample_name}")
            return np.array([]), np.array([])
        wavelengths, average_intensities = zip(*data)
        return np.array(wavelengths), np.array(average_intensities)

class c:
    @staticmethod
    def fcp(
        wavelengths,
        intensities,
        nist_wavelengths,
        nist_element,
        nist_num,
        nist_acc,
        prominence=None,
        height=0.1,
        tolerance=1,
        width=None,
        distance=None,
        min_snr=None,
    ):
        prominence = float(prominence) if prominence is not None else None
        height = float(height) if height is not None else 0.1
        peaks, _ = find_peaks(
            intensities,
            prominence=prominence,
            height=height,
            distance=distance,
            width=width,
        )
        peak_wavelengths = wavelengths[peaks]
        peak_intensities = intensities[peaks]
        if min_snr is not None:
            noise = np.mean(
                np.concatenate(
                    [
                        intensities[wavelengths < min(peak_wavelengths) - tolerance],
                        intensities[wavelengths > max(peak_wavelengths) + tolerance],
                    ]
                )
            )
            snr = peak_intensities / noise
            peaks = peaks[snr >= min_snr]
            peak_wavelengths = wavelengths[peaks]
            peak_intensities = intensities[peaks]
        closest_peaks = []
        for i, peak_wl in enumerate(peak_wavelengths):
            distances = np.abs(nist_wavelengths - peak_wl)
            sorted_indices = np.argsort(distances)[:20]
            for idx in sorted_indices:
                if distances[idx] < tolerance and nist_num[idx] in [1, 2]:
                    closest_peaks.append(
                        (
                            peak_wl,
                            peak_intensities[i],
                            nist_wavelengths[idx],
                            nist_element[idx],
                            nist_num[idx],
                            nist_acc[idx],
                            distances[idx],
                        )
                    )
        return closest_peaks

    @staticmethod
    @staticmethod
    def pltspec(
            sample_name,
            wavelengths,
            intensities,
            segment_size,
            nist_wavelengths,
            nist_element,
            nist_num,
            nist_acc,
            pdf_filename,
            prominence=None,
            height=0.1,
            savgol_window=11,
            savgol_poly=3,
            min_snr=None,
            apply_smoothing=True,
            output_text_file=None
    ):
        # Membuka file teks untuk menulis hasil pilihan
        with open(output_text_file, "w") as output_file:
            output_file.write(f"Spektrum Sampel: {sample_name}\n")
            output_file.write(f"Segment Size: {segment_size} nm\n\n")
            output_file.write("Puncak terpilih:\n")

        if apply_smoothing and savgol_window is not None and savgol_poly is not None:
            intensities_smooth = savgol_filter(intensities, savgol_window, savgol_poly)
        else:
            intensities_smooth = intensities

        num_segments = int(np.ceil((wavelengths[-1] - wavelengths[0]) / segment_size))
        pdf_pages = PdfPages(pdf_filename)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wavelengths, intensities_smooth, color="black", linewidth=0.4)
        ax.grid(True)
        ax.set_xlabel("Panjang Gelombang (nm)")
        ax.set_ylabel("Intensitas")
        ax.set_title(f"Spektrum Penuh Sampel {sample_name}")
        pdf_pages.savefig(fig)
        plt.close(fig)

        for i in range(num_segments):
            segment_start = wavelengths[0] + i * segment_size
            segment_end = segment_start + segment_size
            mask = (wavelengths >= segment_start) & (wavelengths <= segment_end)
            wavelengths_zoomed = wavelengths[mask]
            intensities_zoomed = intensities_smooth[mask]
            if len(wavelengths_zoomed) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                wavelengths_zoomed, intensities_zoomed, color="black", linewidth=0.4
            )
            ax.grid(True)
            closest_peaks = c.fcp(
                wavelengths_zoomed,
                intensities_zoomed,
                nist_wavelengths,
                nist_element,
                nist_num,
                nist_acc,
                prominence=prominence,
                height=height,
                min_snr=min_snr,
            )
            closest_peaks = sorted(closest_peaks, key=lambda x: x[1], reverse=True)
            selected_peaks = []
            for j, (
                    peak_wl,
                    peak_int,
                    nist_wl,
                    element,
                    ion_stage,
                    acc,
                    distance,
            ) in enumerate(closest_peaks):
                print(
                    f"{j + 1}: {element} {ion_stage} @ {nist_wl:.6f} nm (Δλ = {distance:.6f} nm, ACC = {acc}, peakin {peak_int:.6f} )"
                )
                if (j + 1) % 20 == 0 or j == len(closest_peaks) - 1:
                    choice = input(
                        f"Pilih puncak untuk {peak_wl:.6f} nm (masukkan nomor atau tekan Enter untuk lewati): "
                    )
                    if choice.isdigit():
                        selected_idx = int(choice) - 1
                        if 0 <= selected_idx < len(closest_peaks):
                            selected_peaks.append(closest_peaks[selected_idx])
                            # Tuliskan pilihan ke file teks
                            with open(output_text_file, "a") as output_file:
                                output_file.write(
                                    f"{closest_peaks[selected_idx][3]} {closest_peaks[selected_idx][4]} {closest_peaks[selected_idx][2]:.6f} {closest_peaks[selected_idx][5]})\n"
                                )
                        else:
                            print("Pilihan tidak valid, puncak otomatis akan dipilih.")
                            auto_selected_peak = min(
                                closest_peaks[j - 19: j + 1],
                                key=lambda x: (c.accp(x[6], x[5]), x[-1]),
                            )
                            selected_peaks.append(auto_selected_peak)
                            print(
                                f"Puncak otomatis dipilih: {auto_selected_peak[3]} {auto_selected_peak[4]} {auto_selected_peak[2]:.6f} nm "
                            )
                            # Tuliskan pilihan otomatis ke file teks
                            with open(output_text_file, "a") as output_file:
                                output_file.write(
                                    f"{auto_selected_peak[3]} {auto_selected_peak[4]} {auto_selected_peak[2]:.6f} {auto_selected_peak[5]}\n"
                                )
                    else:
                        auto_selected_peak = min(
                            closest_peaks[j - 19: j + 1],
                            key=lambda x: (c.accp(x[6], x[5]), x[-1]),
                        )
                        selected_peaks.append(auto_selected_peak)
                        print(
                            f"Puncak otomatis dipilih: {auto_selected_peak[2]:.6f} nm - {auto_selected_peak[3]} {auto_selected_peak[4]}"
                        )
                        # Tuliskan pilihan otomatis ke file teks
                        with open(output_text_file, "a") as output_file:
                            output_file.write(
                                f"{auto_selected_peak[3]} {auto_selected_peak[4]} {auto_selected_peak[2]:.6f} {auto_selected_peak[5]}\n"
                            )
                    if (j + 1) % 20 == 0:
                        print("Tampilkan 20 puncak terdekat berikutnya...")

            for (
                    peak_wl,
                    peak_int,
                    nist_wl,
                    element,
                    ion_stage,
                    acc,
                    _,
            ) in selected_peaks:
                ion_stage = (
                    "I"
                    if ion_stage == 1
                    else "II" if ion_stage == 2 else str(ion_stage)
                )
                ax.scatter(peak_wl, peak_int, color="red", s=8, marker=".")
                label = f"{element} {ion_stage} {nist_wl:.6f} nm"
                ax.text(
                    peak_wl,
                    peak_int,
                    label,
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    color="blue",
                )
            ax.set_xlabel("Panjang Gelombang (nm)")
            ax.set_ylabel("Intensitas")
            ax.set_title(
                f"Spektrum Sampel {sample_name} pada {segment_start:.2f}-{segment_end:.2f} nm"
            )
            pdf_pages.savefig(fig)
            plt.close(fig)
        pdf_pages.close()
        print(f"Spektrum tersimpan dalam {pdf_filename}")

        print(f"Hasil pilihan puncak disimpan dalam {output_text_file}")
    @staticmethod
    def accp(acc, distance):
        rentang_grade = {
            "AAA": 0.005,
            "AA": 0.01,
            "A+": 0.02,
            "A": 0.03,
            "B+": 0.07,
            "B": 0.10,
            "C+": 0.18,
            "C": 0.25,
        }
        priority_mapping = {
            "AAA": 1,
            "AA": 2,
            "A+": 3,
            "A": 4,
            "B+": 5,
            "B": 6,
            "C+": 7,
            "C": 8,
            "D+": 9,
            "D": 10,
            "E": 11,
        }
        if acc not in priority_mapping:
            return 12
        if distance <= rentang_grade.get(acc, float("inf")):
            return priority_mapping[acc]
        else:
            return priority_mapping[acc] + 5

class k:

    @staticmethod
    def ch(closest_peaks):
        selected_peaks = []
        max_peaks = 20
        num_peaks = len(closest_peaks)
        num_selected_peaks = 0
        sorted_peaks = sorted(closest_peaks, key=lambda x: x[1], reverse=True)
        for i, (peak_wl, peak_int, _, _, _, _, _) in enumerate(sorted_peaks):
            if num_selected_peaks >= max_peaks:
                break
            print(
                f"Memilih puncak ke-{i + 1} ({peak_wl:.6f} nm) dan 20 puncak terdekat..."
            )
            start_idx = i * max_peaks
            end_idx = start_idx + max_peaks
            current_peak_set = sorted_peaks[start_idx:end_idx]
            for j, (
                peak_wl,
                peak_int,
                nist_wl,
                element,
                ion_stage,
                acc,
                distance,
            ) in enumerate(current_peak_set):
                print(
                    f"{j + 1}: {element} {ion_stage} @ {nist_wl:.6f} nm (Δλ = {distance:.6f} nm, ACC = {acc})"
                )
            choice = input(
                "Masukkan nomor puncak untuk digunakan dalam kalibrasi (atau ketik 'selesai' untuk berhenti): "
            ).strip()
            if choice.lower() == "selesai":
                break
            elif choice.isdigit():
                selected_idx = int(choice) - 1
                if 0 <= selected_idx < len(current_peak_set):
                    selected_peaks.append(current_peak_set[selected_idx])
                    num_selected_peaks += 1
                    print(f"Puncak dipilih: {current_peak_set[selected_idx][2]:.6f} nm")
                else:
                    print(
                        "Nomor puncak tidak valid. Silakan pilih nomor puncak yang valid."
                    )
            else:
                print(
                    "Input tidak valid. Silakan masukkan nomor puncak atau ketik 'selesai'."
                )
        print("Puncak yang dipilih untuk kalibrasi:")
        for peak in selected_peaks:
            print(
                f"Puncak pada {peak[0]:.6f} nm (NIST: {peak[2]:.6f} nm, Elemen: {peak[3]}, ACC: {peak[5]})"
            )
        return selected_peaks

    @staticmethod
    def fk(selected_peaks):
        if not selected_peaks:
            print("Tidak ada puncak terpilih untuk kalibrasi.")
            return None
        eksperimen_wavelengths = np.array(
            [peak_wl for peak_wl, _, _, _, _, _, _ in selected_peaks]
        )
        nist_wavelengths = np.array(
            [nist_wl for _, _, nist_wl, _, _, _, _ in selected_peaks]
        )
        def residuals(factor, measured, reference):

            return (measured * factor) - reference
        result = least_squares(
            residuals, x0=[1.0], args=(eksperimen_wavelengths, nist_wavelengths)
        )
        calibration_factor = result.x[0]
        print(f"Faktor kalibrasi: {calibration_factor:.6f}")
        return calibration_factor

def main():
    db_path = "tanah_vulkanik.db"
    nist_db_path = "data1.db"
    nist_wavelengths, nist_elements, nist_nums, nist_acc = baca.nist(nist_db_path)
    while True:
        sample_name_input = input("Sampel : ")
        sample_name = f"S{sample_name_input}" if sample_name_input.strip() else "S1"
        try:
            lower_bound = 100* float(input("Masukkan batas bawah spektrum (nm): ") or 200)
            upper_bound = 100*float(input("Masukkan batas atas spektrum (nm): ") or 900)
        except ValueError:
            print("Batas spektrum harus berupa angka.")
            continue
        wavelengths, average_intensities = baca.spec(
            db_path, sample_name, lower_bound, upper_bound
        )
        if len(wavelengths) > 0 and len(average_intensities) > 0:
            try:
                use_existing_calibration = (
                    input("Apakah Anda sudah memiliki faktor kalibrasi? (y/n): ")
                    .strip()
                    .lower()
                )
                if use_existing_calibration == "y":
                    calibration_factor = (
                        float(
                            input(
                                "Masukkan faktor kalibrasi yang sudah ada (dalam nm): "
                            ).strip()
                        )
                        or 1.000016
                    )
                else:

                    print(
                        "Memilih 20 puncak terdekat dari puncak tertinggi eksperimen..."
                    )

                    closest_peaks = c.fcp(
                        wavelengths,
                        average_intensities,
                        nist_wavelengths,
                        nist_elements,
                        nist_nums,
                        nist_acc,
                        prominence=None,
                        height=0.1,
                        tolerance=1,
                        width=None,
                        distance=None,
                        min_snr=None,
                    )
                    selected_peaks = k.ch(closest_peaks)
                    calibration_factor = k.fk(selected_peaks)
                if calibration_factor is not None:
                    wavelengths = wavelengths * calibration_factor
                    print(
                        f"Spektrum telah dikalibrasi dengan faktor: {calibration_factor:.6f} nm"
                    )
                segment_size = float(input("Masukkan ukuran segmen (nm): ") or 200)
                min_snr_input = input("Masukkan nilai minimum SNR (default: None): ")
                min_snr = float(min_snr_input) if min_snr_input.strip() else 3
                prominence_input = input(
                    "Masukkan nilai prominence (atau abaikan dengan enter): "
                )
                prominence = (
                    float(prominence_input) if prominence_input.strip() else None
                )
                height_input = input("Masukkan nilai height (default: 0.1): ")
                height = float(height_input) if height_input.strip() else 100000
                apply_smoothing = input("Apakah ingin menggunakan smoothing? (y/n): ").strip().lower() == 'y'
                if apply_smoothing:
                    savgol_window_input = input(
                        "Masukkan ukuran jendela Savitzky-Golay (default: 11): "
                    )
                    savgol_window = (
                        int(savgol_window_input) if savgol_window_input.strip() else 11
                    )
                    savgol_poly_input = input(
                        "Masukkan orde polinomial Savitzky-Golay (default: 3): "
                    )
                    savgol_poly = int(savgol_poly_input) if savgol_poly_input.strip() else 3
                else:
                    savgol_window = None
                    savgol_poly = None
                tz = pytz.timezone('Asia/Jakarta')
                wktu = datetime.fromtimestamp(time.time(), tz).strftime("%m%d-%H%M")
                pdf_filename = (
                    f"{sample_name}({upper_bound}-{lower_bound})({height})({wktu}).pdf"
                )
                otpe = f"{sample_name}({upper_bound}-{lower_bound})({height})({wktu}).txt"
                c.pltspec(
                    sample_name,
                    wavelengths,
                    average_intensities,
                    segment_size,
                    nist_wavelengths,
                    nist_elements,
                    nist_nums,
                    nist_acc,
                    pdf_filename,
                    prominence,
                    height,
                    savgol_window,
                    savgol_poly,
                    apply_smoothing=apply_smoothing,
                    output_text_file=otpe,

                )
            except ValueError as e:
                print(f"Input tidak valid: {e}")
        else:
            print("Data spektrum tidak ditemukan untuk kombinasi yang diberikan.")
        if input("Apakah Anda ingin keluar? (y/n): ").strip().lower() == "y":
            break

if __name__ == "__main__":
  main()
