import pandas as pd
import os

# Paths to the dataset subfolders
base_path= r'D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Project\Dataset\data'
folders = ['train', 'test', 'valid']

# Function to check missing values
def check_missing_values(df, name):
    missing_values = df.isnull().sum()
    print(f"\nMissing values in {name} dataset:")
    print(missing_values[missing_values > 0])  # Only print columns with missing values

# Check for missing values in each dataset
check_missing_values(pd.read_csv('D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Project\Dataset\Data\train\_annotations.csv'), "Train")
check_missing_values(pd.read_csv('D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Project\Dataset\Data\test\_annotations.csv'), "Test")
check_missing_values(pd.read_csv('D:\Documents\KULIAH\Semester 5\Studi Independen\Project Capstone\Project\Dataset\Data\valid\_annotations.csv'), "Valid")

for folder in folders:
    # Load the CSV file for the current folder
    csv_file_path = os.path.join(base_path, folder, '_annotations.csv')
    data = pd.read_csv(csv_file_path)

    # Menghapus baris duplikat berdasarkan kolom 'filename'
    data = data.drop_duplicates(subset='filename', keep='first').reset_index(drop=True)

    # Get the list of image filenames from the CSV
    csv_filenames = data['filename'].tolist()

    # Specify the directory where the images are stored
    image_directory = os.path.join(base_path, folder)

    # Get the list of actual filenames in the directory
    actual_filenames = os.listdir(image_directory)

    # Find filenames in the CSV that are not present in the image directory
    missing_images = [filename for filename in csv_filenames if filename not in actual_filenames]

    # Print the missing images
    if missing_images:
        print(f"Missing images in {folder}:", missing_images)

        # Remove entries for missing images from the DataFrame
        data = data[~data['filename'].isin(missing_images)]

    # Save the cleaned DataFrame to a new CSV file in the respective folder
    cleaned_csv_file_path = os.path.join(base_path, folder, 'cleaned_annotations.csv')
    data.to_csv(cleaned_csv_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_csv_file_path}")

    # If no missing images, still save the cleaned data
    if not missing_images:
        print(f"All images in {folder} are accounted for.")