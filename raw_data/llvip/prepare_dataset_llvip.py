import os
import shutil
import xml.etree.ElementTree as ET

def copy_jpg_for_keras_flow_from_directory_adjusted():
    base_path = os.path.join(os.getcwd(), 'raw_data')
    annotations_path = os.path.join(base_path, 'Annotations')
    dataset_path = os.path.join(os.getcwd(), '../dataset')

    # Subdirectory and category configuration
    subdirs = ['infrared']
    categories = ['train', 'test']
    classes = ['person', 'not-person']

    # Initialize counters
    total_copied = 0
    copied_to_person = 0
    copied_to_not_person = 0

    # Ensure target directories exist
    for subdir in subdirs:
        for category in categories:
            for class_ in classes:
                os.makedirs(os.path.join(dataset_path, category, class_), exist_ok=True)

    for file in os.listdir(annotations_path):
        if file.endswith(".xml"):
            has_person = False
            file_path = os.path.join(annotations_path, file)
            tree = ET.parse(file_path)
            root = tree.getroot()

            for object_name in root.findall(".//object/name"):
                if object_name.text == "person":
                    has_person = True
                    break

            target_class = 'person' if has_person else 'not-person'
            jpg_filename = os.path.splitext(file)[0] + '.jpg'

            # Copy the .jpg file from specified subdirectories
            for subdir in subdirs:
                for category in categories:
                    source_jpg_path = os.path.join(base_path, subdir, category, jpg_filename)
                    if os.path.exists(source_jpg_path):
                        target_jpg_path = os.path.join(dataset_path, category, target_class, jpg_filename)
                        shutil.copy(source_jpg_path, target_jpg_path)

                        # Update counters
                        total_copied += 1
                        if has_person:
                            copied_to_person += 1
                        else:
                            copied_to_not_person += 1

    # Print summary
    print(f"Total files copied: {total_copied}")
    print(f"Files copied to 'person': {copied_to_person}")
    print(f"Files copied to 'not-person': {copied_to_not_person}")

print("Copying files...")
copy_jpg_for_keras_flow_from_directory_adjusted()
