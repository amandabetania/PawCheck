import os
import cv2
import numpy as np
from tqdm import tqdm

# Path dataset
base_path_train = r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Program\dataset\train\Healthy'
# base_path_test = r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Program\dataset\test\Healthy'
# base_path_valid = r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Program\dataset\valid\Healthy'

# Output path untuk menyimpan gambar yang sudah diolah
output_base_train = r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Program\dataset\train\Healthy'
# output_base_test = r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Program\dataset\test\Healthy'
# output_base_valid = r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Program\dataset\valid\Healthy'

# Fungsi untuk memastikan direktori output ada
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Fungsi untuk memproses gambar
def process_image(image_path):
    # Membaca gambar
    img = cv2.imread(image_path)

    # Menghapus noise dengan Gaussian Blur
    denoised_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Konversi ke ruang warna YUV untuk perbaikan kecerahan
    yuv_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2YUV)

    # Periksa tingkat kecerahan
    avg_brightness = np.mean(yuv_img[:, :, 0])  # Channel Y

    # Jika rata-rata kecerahan rendah, gunakan CLAHE
    if avg_brightness < 120:  # Threshold kecerahan (adjust sesuai kebutuhan)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv_img[:, :, 0] = clahe.apply(yuv_img[:, :, 0])

    # Konversi kembali ke ruang warna BGR
    processed_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    return processed_img

# Fungsi untuk memproses dataset
def process_dataset(input_base_path, output_base_path):
    for root, dirs, files in os.walk(input_base_path):
        for file in tqdm(files, desc=f"Processing {root.split('/')[-1]}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_base_path)
                output_path = os.path.join(output_base_path, relative_path)

                # Pastikan direktori output ada
                ensure_dir(os.path.dirname(output_path))

                # Proses gambar
                processed_img = process_image(input_path)

                # Simpan gambar yang telah diolah
                cv2.imwrite(output_path, processed_img)

# Memproses dataset train, test, dan valid
process_dataset(base_path_train, output_base_train)
# process_dataset(base_path_test, output_base_test)
# process_dataset(base_path_valid, output_base_valid)
