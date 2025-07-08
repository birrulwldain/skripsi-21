import os

# --- Konfigurasi ---
# Ekstensi file yang ingin diperiksa
TARGET_EXTENSIONS = ('.tex', '.bib', '.cls', '.txt', '.md')
# Folder yang ingin diabaikan
IGNORED_DIRS = ('.git', '.github', '.vscode', '__pycache__')
# Batas kedalaman pencarian subfolder
MAX_DEPTH = 3

# --- Logika Skrip ---
def check_file_encoding(filepath):
    """Mencoba membaca file dengan encoding UTF-8. 
    Mengembalikan True jika berhasil, False jika gagal."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False
    except Exception:
        # Menangani file yang tidak bisa dibaca, dll.
        return True # Asumsikan benar jika bukan error encoding

def main():
    """Fungsi utama untuk memindai direktori dan mencetak laporan."""
    problem_files = []
    checked_count = 0
    start_path = '.'
    
    # Menghitung kedalaman awal untuk perbandingan
    start_depth = start_path.rstrip(os.path.sep).count(os.sep)

    print(f"--- Memulai Pengecekan Encoding File (Maksimal {MAX_DEPTH} Subfolder) ---")

    # Memindai semua file dan folder dari direktori saat ini
    for root, dirs, files in os.walk(start_path, topdown=True):
        # Mengabaikan folder yang ada di dalam IGNORED_DIRS
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        # Memeriksa dan membatasi kedalaman
        current_depth = root.rstrip(os.path.sep).count(os.sep)
        if current_depth - start_depth >= MAX_DEPTH:
            # Jika kedalaman tercapai, jangan masuk ke subfolder lebih dalam lagi
            dirs[:] = []

        for filename in files:
            if filename.endswith(TARGET_EXTENSIONS):
                filepath = os.path.join(root, filename)
                checked_count += 1
                if not check_file_encoding(filepath):
                    problem_files.append(filepath)

    print(f"Selesai. Memeriksa total {checked_count} file.")
    print("-" * 45)

    # Mencetak hasil laporan
    if not problem_files:
        print("✅ Semua file target dalam jangkauan sudah dalam format UTF-8.")
    else:
        print("❌ Ditemukan file dengan encoding BUKAN UTF-8:")
        for path in problem_files:
            print(f"   -> {path}")
        print("\nHarap buka file di atas di VS Code dan simpan ulang (Save with Encoding) ke format UTF-8.")

if __name__ == "__main__":
    main()