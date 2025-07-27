import os
import sys

# Menambahkan path 'engine' agar dapat mengimpor simul1
script_dir = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(script_dir) 
sys.path.append(engine_path)

try:
    # Mengimpor fungsi utama dari modul simulasi Anda
    from simul1 import run_simulation, SIMULATION_CONFIG
except ImportError:
    print("Gagal mengimpor 'run_simulation' dari 'simul1.py'.")
    print("Pastikan skrip ini dijalankan dari direktori 'engine' Anda.")
    sys.exit(1)

if __name__ == "__main__":
    # --- Konfigurasi Plot ---
    
    # 1. Tentukan elemen dan konsentrasinya (total harus 100%)
    elements_to_plot = [
        ('Al', 50.0),
        ('Ca', 50.0)
    ]
    
    # 2. Tentukan kondisi plasma (sama seperti pada tabel sebelumnya)
    TEMPERATURE = 8000.0         # Suhu dalam Kelvin
    ELECTRON_DENSITY = 1.0e17    # Densitas elektron dalam cm^-3
    
    # 3. (Opsional) Tentukan rentang panjang gelombang
    # Biarkan default untuk melihat keseluruhan spektrum
    # SIMULATION_CONFIG["wl_range"] = (350, 450) 

    # --- Eksekusi Simulasi dan Plotting ---
    
    # Menyiapkan path ke file data, sama seperti di simul1.py
    data_dir = os.path.join(script_dir, "data")
    nist_data_file = os.path.join(data_dir, "nist_data(1).h5")
    atomic_data_file = os.path.join(data_dir, "atomic_data1.h5")
    
    print("="*50)
    print("Membuat plot spektrum campuran Al (50%) dan Ca (50%)...")
    print(f"Suhu: {TEMPERATURE} K, Densitas Elektron: {ELECTRON_DENSITY:.1e} cm⁻³")
    print("="*50)
    
    # Memanggil fungsi inti dari simul1.py untuk melakukan semua pekerjaan
    run_simulation(
        selected_elements=elements_to_plot,
        temperature=TEMPERATURE,
        electron_density=ELECTRON_DENSITY,
        nist_path=nist_data_file,
        atomic_data_path=atomic_data_file
    )