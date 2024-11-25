from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pandas as pd

# Fungsi untuk membersihkan dataset
def clean_dataset(df):
    # Normalisasi/Standarisasi
    scaler = StandardScaler()
    df[['width', 'height']] = scaler.fit_transform(df[['width', 'height']])

    # Encoding kategori
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])

    return df

cleaned_datasets = {}
base_path = r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Program\dataset'
folders = ['train', 'test', 'valid']

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    file_path = os.path.join(folder_path, 'cleaned_annotations.csv')  # Path ke file cleaned_annotation.csv
    if os.path.isfile(file_path):  # Cek apakah file ada
        print(f"Memproses file: {file_path}")  # Output proses file

        # Membaca file CSV
        df = pd.read_csv(file_path)

        # Menerapkan pembersihan dataset
        cleaned_df = clean_dataset(df)
        print(f"Dataset pada folder '{folder}' berhasil dibersihkan.")  # Output sukses

        # Menyimpan dataset yang sudah dibersihkan
        cleaned_datasets[folder] = cleaned_df
    else:
        print(f"File {file_path} tidak ditemukan. Skipping.")  # Output jika file tidak ditemukan

print("Semua folder selesai diproses!")  # Output setelah semua folder selesai