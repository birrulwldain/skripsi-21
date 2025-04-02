import sqlite3

# Fungsi untuk mendapatkan data tambahan dari database NIST berdasarkan panjang gelombang
def get_additional_data(wavelength):
    # Koneksi ke database NIST
    conn = sqlite3.connect('data1.db')  # Pastikan nama database sesuai
    cursor = conn.cursor()

    # Query untuk mengambil data Ei dan Ek berdasarkan panjang gelombang
    query = """
            SELECT 
                "obs_wl_air(nm)", 
                "element", 
                "Ei(eV)", 
                "Ek(eV)", 
                "acc"
            FROM spectrum_data
            WHERE "obs_wl_air(nm)" BETWEEN ? AND ? AND "sp_num" IN (1, 2)
            """

    # Set toleransi pencocokan (misalnya 0.001 untuk toleransi)
    tolerance = 0.001
    cursor.execute(query, (wavelength - tolerance, wavelength + tolerance))

    result = cursor.fetchone()
    conn.close()
    return result

# Data awal (seperti yang kamu miliki)
initial_data = [
    {"element": "Dy", "wavelength": 396.839000, "acc": "nan"},
    {"element": "Th", "wavelength": 396.150300, "acc": "nan"},
    {"element": "Tc", "wavelength": 393.370500, "acc": "nan"},
    {"element": "Ti", "wavelength": 279.539900, "acc": "nan"},
    {"element": "Cr", "wavelength": 280.237530, "acc": "nan"},
    {"element": "V", "wavelength": 394.393700, "acc": "C+"},
    {"element": "W", "wavelength": 288.160600, "acc": "nan"},
    {"element": "W", "wavelength": 309.267600, "acc": "nan"},
    {"element": "Cr", "wavelength": 390.564390, "acc": "nan"},
    {"element": "Fe", "wavelength": 285.201460, "acc": "nan"},
    {"element": "Al", "wavelength": 422.681300, "acc": "C+"},
]

# Fungsi untuk melengkapi data dan menuliskannya ke file teks
def complete_and_write_data(data):
    with open("completed_data.txt", "w") as file:
        for entry in data:
            additional_data = get_additional_data(entry["wavelength"])
            if additional_data:
                wavelength, element, Ei, Ek, acc_db = additional_data
                # Pastikan wavelength dalam bentuk float sebelum diformat
                wavelength = float(wavelength)
                Ei = Ei if Ei is not None else "Ei(eV)"
                Ek = Ek if Ek is not None else "Ek(eV)"
                # Menulis data yang telah dilengkapi ke file teks
                file.write(f"{element} {wavelength:.6f} {Ei} {Ek} {acc_db}\n")
            else:
                # Jika data tambahan tidak ditemukan, tulis data awal
                Ei = "Ei(eV)"
                Ek = "Ek(eV)"
                file.write(f"{entry['element']} {entry['wavelength']:.6f} {Ei} {Ek} {entry['acc']}\n")

# Lengkapi data dan tulis ke file
complete_and_write_data(initial_data)