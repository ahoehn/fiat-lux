import json
import os
import shutil

def execution(file_path, source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Counters
    copied_files_count = 0
    frames_with_only_person = 0
    frames_with_only_non_person = 0
    frames_with_person_and_other = 0

    # Read the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Analyze the frames according to the defined categories
    for frame in data['frames']:
        labels = [label for annotation in frame['annotations'] for label in annotation['labels']]
        unique_labels = set(labels)

        # Check if the frame belongs to one of the categories
        if unique_labels == {'person'}:
            frames_with_only_person += 1
        elif 'person' not in unique_labels:
            frames_with_only_non_person += 1
            videoId = frame['videoMetadata']['videoId']
            frameIndex = frame['videoMetadata']['frameIndex']
            datasetFrameId = frame['datasetFrameId']

            # Create filename according to the specified pattern
            filename = f"video-{videoId}-frame-{frameIndex:06d}-{datasetFrameId}.jpg"

            # Construct source and target file paths
            source_file_path = os.path.join(source_dir, filename)
            target_file_path = os.path.join(target_dir, filename)

            # Check if the file exists in the source directory and copy it
            if os.path.exists(source_file_path):
                shutil.copy(source_file_path, target_file_path)
                copied_files_count += 1
        else:
            frames_with_person_and_other += 1

    # Count total number of frames
    total_frames = len(data['frames'])

    # Print the results
    print(f"Frames with only 'person' as label items: {frames_with_only_person}")
    print(f"Frames with only label items different to 'person': {frames_with_only_non_person}")
    print(f"Frames with 'person' and any other item as label item: {frames_with_person_and_other}")
    print(f"Number of files copied: {copied_files_count}")
    print(f"Total number of frames: {total_frames}")


# use the training data as train data and split it later in the python notebook into train and validation data
file_path = 'raw_data/images_thermal_train/index.json'
source_dir = 'raw_data/images_thermal_train/data'
target_dir = '../dataset/train/not-person'
execution(file_path, source_dir, target_dir)

# use the validation data as test data (because no dedicated validation data is provided)
file_path = 'raw_data/images_thermal_val/index.json'
source_dir = 'raw_data/images_thermal_val/data'
target_dir = '../dataset/test/not-person'
execution(file_path, source_dir, target_dir)