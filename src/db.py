import sqlite3
import pandas as pd


def clean_column_values(df):
    """Membersihkan nilai kolom dari tanda kutip ganda dan simbol ="""
    for col in df.columns:
        df[col] = df[col].replace({'="': '', '"': '', '=': ''}, regex=True)
    return df


def create_connection(db_file):
    """Membuat koneksi ke database SQLite"""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Koneksi ke database {db_file} berhasil.")
    except sqlite3.Error as e:
        print(e)
    return conn


def create_table_from_csv(conn, csv_file, table_name):
    """Membuat tabel dan memasukkan data dari file CSV"""
    # Membaca file CSV
    df = pd.read_csv(csv_file, delimiter=',', quotechar='"')

    # Membersihkan format data yang berasal dari Excel seperti ="value"
    df = clean_column_values(df)

    # Menyimpan DataFrame ke tabel SQLite
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Tabel {table_name} berhasil dibuat dan data diimpor dari {csv_file}.")


def check_table_and_columns(conn, table_name):
    """Memeriksa tabel yang ada dan header kolom di dalam database"""
    cursor = conn.cursor()

    # Memeriksa tabel yang ada di database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tabel yang ada di dalam database:")
    for table in tables:
        print(f"- {table[0]}")

    # Memeriksa kolom dari tabel tertentu
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print(f"Header atau kolom dari tabel {table_name}:")
    for column in columns:
        print(f"- {column[1]} (type: {column[2]})")


def main():
    database = "data1.db"  # Nama database yang ingin dibuat
    csv_file = "2-4.csv"  # Ganti dengan path file CSV Anda
    table_name = "spectrum_data"  # Nama tabel di database

    # Membuat koneksi ke database
    conn = create_connection(database)

    if conn is not None:
        # Membuat tabel dan mengimpor data dari CSV
        create_table_from_csv(conn, csv_file, table_name)

        # Memeriksa tabel dan kolom yang ada di database
        check_table_and_columns(conn, table_name)

        conn.close()
    else:
        print("Error! Tidak dapat membuat koneksi ke database.")


if __name__ == "__main__":
    main()