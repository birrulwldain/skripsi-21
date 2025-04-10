import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares, curve_fit
from scipy.spatial import procrustes

def read_nist_from_db(db_path):
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
        WHERE "element" = 'Cu' AND "sp_num" IN (1, 2) AND "acc" IN ('AA', 'A+', 'A')
        """
    )
    data = cursor.fetchall()
    conn.close()

    if not data:
        print("Tidak ada data Cu ditemukan di database.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    nist_wavelengths, nist_elements, nist_nums, nist_acc = zip(*data)
    nist_wavelengths = np.array([float(w) if w else np.nan for w in nist_wavelengths])
    nist_acc = np.array([str(a) if a else np.nan for a in nist_acc])
    return nist_wavelengths, nist_elements, nist_nums, nist_acc

def read_asc_file(file_path):
    """Membaca data spektrum dari file ASC yang tidak memiliki header."""
    data = np.loadtxt(file_path, delimiter='\t')
    spectrum_wavelengths = data[:, 0]
    spectrum_intensities = data[:, 1]
    return spectrum_wavelengths, spectrum_intensities

def find_closest_peaks(wavelengths, intensities, nist_wavelengths, nist_element, nist_num, nist_acc, tolerance=0.1):
    peaks, _ = find_peaks(intensities, height=0.1)
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

def calculate_calibration_factor(closest_peaks):
    """Menghitung faktor kalibrasi menggunakan metode sederhana rata-rata rasio."""
    measured_peaks = []
    reference_peaks = []

    for peak_wl, peak_int, nist_wl, element, ion_stage, acc, _ in closest_peaks:
        measured_peaks.append(peak_wl)
        reference_peaks.append(nist_wl)

    measured_peaks = np.array(measured_peaks)
    reference_peaks = np.array(reference_peaks)

    calibration_factors = reference_peaks / measured_peaks
    average_calibration_factor = np.mean(calibration_factors)

    return average_calibration_factor

def calibrate_using_least_squares(measured_peaks, reference_peaks):
    """Menghitung faktor kalibrasi menggunakan metode Least Squares."""

    def residuals(factor, measured, reference):
        return (measured * factor) - reference

    result = least_squares(residuals, x0=[1.0], args=(measured_peaks, reference_peaks))
    return result.x[0]

def calibrate_using_icp(measured_peaks, reference_peaks):
    """Menghitung faktor kalibrasi menggunakan metode Iterative Closest Point (ICP)."""
    mtx1, mtx2, disparity = procrustes(reference_peaks.reshape(-1, 1), measured_peaks.reshape(-1, 1))
    icp_factor = np.mean(mtx2) / np.mean(mtx1)
    return icp_factor

def nonlinear_function(x, a, b):
    return a * x + b

def calibrate_using_nonlinear(measured_peaks, reference_peaks):
    """Menghitung faktor kalibrasi menggunakan metode Nonlinear Regression."""
    popt, _ = curve_fit(nonlinear_function, measured_peaks, reference_peaks)
    return popt  # popt[0] is the factor 'a', popt[1] is the intercept 'b'


def plot_spectra_and_nist_to_pdf(spectrum_wavelengths, spectrum_intensities, nist_wavelengths, nist_elements, nist_nums,
                                 nist_acc, pdf_filename, calibrated_spectra=None,
                                 average_calibration_factor=None, calibration_factor_least_squares=None,
                                 calibration_factor_icp=None, a=None):
    """Memvisualisasikan spektrum dan menandai puncak yang sesuai dengan data NIST dalam PDF."""
    with PdfPages(pdf_filename) as pdf:
        plt.figure(figsize=(12, 6))

        # Plot spektrum dari file ASC
        plt.plot(spectrum_wavelengths, savgol_filter(spectrum_intensities, 11, 3), color='black', label='Spektrum ASC',
                 linewidth=0.4)

        # Cari dan plot puncak yang sesuai dengan NIST
        closest_peaks = find_closest_peaks(spectrum_wavelengths, spectrum_intensities, nist_wavelengths, nist_elements,
                                           nist_nums, nist_acc)

        for peak_wl, peak_int, nist_wl, element, ion_stage, acc, _ in closest_peaks:
            ion_stage = "I" if ion_stage == 1 else "II" if ion_stage == 2 else str(ion_stage)
            plt.scatter(peak_wl, peak_int, color="red", s=5, marker=".")
            plt.text(peak_wl, peak_int, f"{element} {ion_stage} {nist_wl:.6f} nm", fontsize=5, ha="center", va="bottom",
                     rotation=90, color="blue")

        plt.xlabel('Panjang Gelombang (nm)')
        plt.ylabel('Intensitas')
        plt.title('Spektrum Tembaga Murni dengan Puncak NIST')
        plt.legend()
        plt.grid(True)

        # Simpan halaman pertama (Spektrum ASC dan puncak NIST) ke file PDF
        pdf.savefig()
        plt.close()

        # Plotkan spektrum terkalibrasi untuk setiap metode
        for label, calib_spectrum in calibrated_spectra.items():
            plt.figure(figsize=(12, 6))
            plt.plot(spectrum_wavelengths, savgol_filter(calib_spectrum, 11, 3),
                     label=f'Spektrum Terkalibrasi ({label})')

            # Kalibrasi panjang gelombang untuk label puncak
            if label == 'Rasio Rata-Rata':
                calibration_factor = average_calibration_factor
            elif label == 'Least Squares':
                calibration_factor = calibration_factor_least_squares
            elif label == 'ICP':
                calibration_factor = calibration_factor_icp
            elif label == 'Nonlinear Regression':
                calibration_factor = a

            calibrated_peak_wavelengths = np.array(
                [peak_wl * calibration_factor for peak_wl, _, _, _, _, _, _ in closest_peaks])

            # Plot dan label puncak yang sesuai pada spektrum terkalibrasi
            for i, peak_wl in enumerate(calibrated_peak_wavelengths):
                if i < len(nist_elements):  # Pastikan indeks tetap dalam rentang
                    plt.scatter(peak_wl, calib_spectrum[np.abs(spectrum_wavelengths - peak_wl).argmin()], color="red",
                                s=5, marker=".")
                    plt.text(peak_wl, calib_spectrum[np.abs(spectrum_wavelengths - peak_wl).argmin()],
                             f"{nist_elements[i]} {nist_nums[i]} {nist_wavelengths[i]:.6f} nm", fontsize=5, ha="center",
                             va="bottom", rotation=90, color="blue")

            plt.xlabel('Panjang Gelombang (nm)')
            plt.ylabel('Intensitas')
            plt.title(f'Spektrum Terkalibrasi dengan {label}')
            plt.legend()
            plt.grid(True)

            # Simpan halaman ke file PDF
            pdf.savefig()
            plt.close()
def main():
    asc_file_path = "Cu_D1us-W50us-ii-3500-acc-5_760 torr-skala 5.asc"
    nist_db_path = "data.db"
    pdf_filename = "spektrum_tembaga_murni_comparison.pdf"

    # Membaca data dari file ASC
    spectrum_wavelengths, spectrum_intensities = read_asc_file(asc_file_path)

    # Membaca data NIST dari database
    nist_wavelengths, nist_elements, nist_nums, nist_acc = read_nist_from_db(nist_db_path)

    if len(nist_wavelengths) == 0:
        print("Tidak ada data NIST yang valid untuk Cu.")
        return

    # Cari puncak terdekat
    closest_peaks = find_closest_peaks(spectrum_wavelengths, spectrum_intensities, nist_wavelengths, nist_elements,
                                       nist_nums, nist_acc)

    # Hitung faktor kalibrasi dengan semua metode
    measured_peaks = np.array([peak_wl for peak_wl, _, nist_wl, _, _, _, _ in closest_peaks])
    reference_peaks = np.array([nist_wl for _, _, nist_wl, _, _, _, _ in closest_peaks])

    # Rasio rata-rata
    average_calibration_factor = calculate_calibration_factor(closest_peaks)
    print(f"Faktor Kalibrasi (Rasio Rata-Rata): {average_calibration_factor:.6f}")

    # Least Squares
    calibration_factor_least_squares = calibrate_using_least_squares(measured_peaks, reference_peaks)
    print(f"Faktor Kalibrasi (Least Squares): {calibration_factor_least_squares:.6f}")

    # Iterative Closest Point (ICP)
    calibration_factor_icp = calibrate_using_icp(measured_peaks, reference_peaks)
    print(f"Faktor Kalibrasi (ICP): {calibration_factor_icp:.6f}")

    # Nonlinear Regression
    a, b = calibrate_using_nonlinear(measured_peaks, reference_peaks)
    print(f"Faktor Kalibrasi (Nonlinear Regression): a={a:.6f}, b={b:.6f}")

    # Terapkan faktor kalibrasi
    calibrated_spectra = {
        'Rasio Rata-Rata': spectrum_intensities * average_calibration_factor,
        'Least Squares': spectrum_intensities * calibration_factor_least_squares,
        'ICP': spectrum_intensities * calibration_factor_icp,
        'Nonlinear Regression': spectrum_intensities * a  # tanpa b untuk kesederhanaan
    }

    # Plot spektrum dan hasil kalibrasi ke PDF
    plot_spectra_and_nist_to_pdf(spectrum_wavelengths, spectrum_intensities, nist_wavelengths, nist_elements,
                                 nist_nums, nist_acc, pdf_filename, calibrated_spectra,
                                 average_calibration_factor, calibration_factor_least_squares,
                                 calibration_factor_icp, a)

if __name__ == "__main__":
    main()

