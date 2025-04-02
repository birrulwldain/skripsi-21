import sqlite3
import pandas as pd


def create_connection(db_file):
    """Membuat koneksi ke database SQLite"""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Koneksi ke database {db_file} berhasil.")
    except sqlite3.Error as e:
        print(e)
    return conn


def count_rows_in_columns(db_file, table_name):
    """Menghitung jumlah baris (non-NULL) di setiap kolom dalam tabel di database SQLite"""
    conn = create_connection(db_file)
    if conn is not None:
        try:
            # Mendapatkan nama kolom dari tabel
            query_columns = f"PRAGMA table_info({table_name});"
            columns_info = pd.read_sql_query(query_columns, conn)
            columns = columns_info['name'].tolist()

            # Menghitung jumlah nilai non-NULL untuk setiap kolom
            counts = {}
            for col in columns:
                # Escape nama kolom dengan tanda kutip ganda untuk nama dengan karakter khusus
                col_escaped = f'"{col}"'
                query_count = f"SELECT COUNT({col_escaped}) AS count FROM {table_name} WHERE {col_escaped} IS NOT NULL;"
                result = pd.read_sql_query(query_count, conn)
                counts[col] = result['count'].values[0]

            # Menampilkan hasil hitungan
            print("Jumlah baris untuk setiap kolom:")
            for col, count in counts.items():
                print(f"{col}: {count}")

        except Exception as e:
            print(f"Terjadi kesalahan saat menghitung kolom: {e}")
        finally:
            conn.close()
    else:
        print("Tidak dapat membuat koneksi ke database.")


def read_first_n_rows(db_file, table_name, n=20):
    """Membaca N baris pertama dari tabel dalam database SQLite"""
    conn = create_connection(db_file)
    if conn is not None:
        try:
            query = f"SELECT * FROM {table_name} LIMIT {n};"
            df = pd.read_sql_query(query, conn)

            # Menampilkan hasil
            print(f"20 baris pertama dari tabel {table_name}:")
            print(df)

        except Exception as e:
            print(f"Terjadi kesalahan saat membaca data: {e}")
        finally:
            conn.close()
    else:
        print("Tidak dapat membuat koneksi ke database.")


def main():
    database = "data1.db"  # Nama database yang ingin dibaca
    table_name = "spectrum_data"  # Nama tabel yang ingin dibaca

    # Menghitung jumlah baris untuk setiap kolom dalam tabel
    count_rows_in_columns(database, table_name)

    # Membaca 20 baris pertama dari tabel
    read_first_n_rows(database, table_name)


if __name__ == "__main__":
    main()