import matplotlib.pyplot as plt
import pandas as pd
from simLIBS import simulation

# Daftar elemen tabel periodik yang ingin disimulasikan
periodic_table = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                  'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                  'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                  'Ga', 'Ge', 'As',  'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                  'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                  'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                  'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                  'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                  'Pa', 'U']

# DataFrame untuk menyimpan semua spektrum
all_data = pd.DataFrame()

# Buat list untuk menyimpan elemen yang gagal diambil spektrumnya
failed_elements = []

# Loop untuk setiap elemen dalam tabel periodik
for element in periodic_table:
    try:
        print(f"Simulating element: {element}")

        # Simulasi LIBS untuk setiap elemen dengan komposisi 100%
        libs = simulation.SimulatedLIBS(
            Te=1.0,
            Ne=10 ** 17,
            elements=[element],
            percentages=[100],
            resolution=1000,
            low_w=200,
            upper_w=1000,
            max_ion_charge=2,
            webscraping="dynamic",
        )

        # Dapatkan spektrum ion untuk elemen tersebut
        ion_spectra = libs.get_ion_spectra()

        # Konversi ke DataFrame dan tambahkan kolom elemen
        df_ion_spectra = pd.DataFrame(ion_spectra)
        df_ion_spectra['element'] = element  # Tambahkan kolom elemen

        # Gabungkan data dengan DataFrame utama
        all_data = pd.concat([all_data, df_ion_spectra], ignore_index=True)

    except KeyError as e:
        print(f"Failed to retrieve data for {element}. Error: {e}")
        failed_elements.append(element)

# Simpan seluruh spektrum ke dalam satu file CSV
all_data.to_csv("output_data/all_elements_ion_spectra.csv", index=False)

# Cetak elemen yang gagal diambil spektrumnya
if failed_elements:
    print(f"Failed elements: {', '.join(failed_elements)}")

# Plot contoh spektrum untuk beberapa elemen (opsional)
sample_elements = ['H', 'Fe', 'O', 'W']
plt.figure(figsize=(10, 6))

for element in sample_elements:
    subset = all_data[all_data['element'] == element]
    plt.plot(subset['wavelength'], subset['intensity'], label=f'Spectrum of {element}')

# Tambahkan detail plot
plt.title('Simulated LIBS Spectrum for Selected Elements')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True)

# Simpan plot (opsional)
plt.savefig("output_data/sample_elements_spectrum_plot.png")

# Tampilkan plot
plt.show()