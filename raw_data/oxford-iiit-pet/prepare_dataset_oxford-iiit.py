import shutil
import os

def process_file(input_file, source_folder, target_folder):
    processed_files = set()
    with open(input_file, 'r') as file:
        for line in file:
            # Extract filename and add .jpg extension
            filename = line.split(' ')[0] + '.jpg'
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # Copy file from source to target if it exists
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
                #print(f'Copied {filename} to {target_folder}')
                processed_files.add(filename)
            else:
                print(f'File {filename} not found in source folder.')

    return processed_files

def list_unprocessed_files(source_folder, processed_files):
    all_files = set(os.listdir(source_folder))
    unprocessed_files = all_files - processed_files
    if unprocessed_files:
        print("The following files in the source folder were not copied:")
        for file in unprocessed_files:
            print(file)
    else:
        print("All files in the source folder were copied.")

# Specify the paths
source_folder = 'raw_data/images/grayscale'
test_target_folder = '../dataset/test/not-person'
trainval_target_folder = '../dataset/train/not-person'

# Keep track of all processed files across both input files
all_processed_files = set()

# Process each file and update the set of processed files
all_processed_files.update(process_file('raw_data/annotations/annotations/test.txt', source_folder, test_target_folder))
all_processed_files.update(process_file('raw_data/annotations/annotations/trainval.txt', source_folder, trainval_target_folder))

# List all files in the source folder that were not copied
list_unprocessed_files(source_folder, all_processed_files)
