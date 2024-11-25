import os
import pandas as pd
import shutil

base_path = r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Program\dataset'

# Path ke folder Dataset
train_dir = os.path.join(base_path, 'train')
valid_dir = os.path.join(base_path, 'valid')
test_dir = os.path.join(base_path, 'test')

# Fungsi untuk membaca CSV hanya jika file ada
def read_annotations(file_path):
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File tidak ditemukan: {file_path}")
        return None

# Membaca file CSV untuk setiap set
train_annotations = read_annotations(os.path.join(train_dir, 'cleaned_annotations.csv'))
valid_annotations = read_annotations(os.path.join(valid_dir, 'cleaned_annotations.csv'))
test_annotations = read_annotations(os.path.join(test_dir, 'cleaned_annotations.csv'))

# Fungsi untuk membersihkan dan memvalidasi data CSV
def clean_annotations(annotations):
    if annotations is not None:
        # Menghapus baris dengan nilai kosong di kolom penting
        annotations = annotations.dropna(subset=['class', 'filename']).drop_duplicates()
        return annotations
    return None

# Fungsi untuk membuat subfolder kelas dan memindahkan gambar
def organize_images(annotations, images_dir, max_images_per_class):
    # Bersihkan data
    annotations = clean_annotations(annotations)

    if annotations is None:
        print("Data anotasi kosong atau tidak valid.")
        return

    # Mengelompokkan berdasarkan kelas
    class_groups = annotations.groupby('class')

    for class_name, group in class_groups:
        # Membatasi gambar yang diambil per kelas
        selected_images = group.sample(min(len(group), max_images_per_class), random_state=42)

        # Membuat folder kelas jika belum ada
        class_dir = os.path.join(images_dir, class_name)
        not_selected_dir = os.path.join(images_dir, "not_selected", class_name)  # Folder untuk gambar yang tidak terpilih
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        if not os.path.exists(not_selected_dir):
            os.makedirs(not_selected_dir)

        # Memindahkan gambar terpilih ke subfolder kelas
        for _, row in selected_images.iterrows():
            filename = row['filename']
            src = os.path.join(images_dir, filename)

            # Memastikan bahwa file ada di folder asal (train)
            if os.path.exists(src):
                # Membuat nama baru dengan menambahkan kelas di depan nama file
                new_filename = f"{class_name}_{filename}"

                dst = os.path.join(class_dir, new_filename)  # Menyimpan dengan nama baru
                shutil.move(src, dst)
            else:
                print(f"File tidak ditemukan: {src}")  # Tambahkan log untuk file yang tidak ditemukan

        # Menangani gambar yang tidak terpilih
        not_selected_images = group[~group['filename'].isin(selected_images['filename'])]

        for _, row in not_selected_images.iterrows():
            filename = row['filename']
            src = os.path.join(images_dir, filename)

            # Memastikan gambar yang tidak dipilih ada di folder asal
            if os.path.exists(src):
                # Membuat nama baru untuk gambar yang tidak terpilih
                new_filename = f"{class_name}_{filename}"

                # Memindahkan gambar yang tidak terpilih ke folder 'not_selected'
                dst = os.path.join(not_selected_dir, new_filename)
                shutil.copy(src, dst)  # Salin file ke folder not_selected
            else:
                print(f"File tidak ditemukan: {src}")  # Tambahkan log untuk file yang tidak ditemukan

        print(f"Pengorganisasian gambar untuk kelas '{class_name}' selesai.")

# Melanjutkan dengan pengorganisasian gambar jika file ditemukan
if train_annotations is not None:
    print("Memulai pengorganisasian gambar untuk dataset 'train'...")
    organize_images(train_annotations, train_dir, max_images_per_class=500)

# if valid_annotations is not None:
#     print("Memulai pengorganisasian gambar untuk dataset 'valid'...")
#     organize_images(valid_annotations, valid_dir, max_images_per_class=100)

# if test_annotations is not None:
#     print("Memulai pengorganisasian gambar untuk dataset 'test'...")
#     organize_images(test_annotations, test_dir, max_images_per_class=20)

# Output akhir
print("Semua dataset telah berhasil diorganisir!")
