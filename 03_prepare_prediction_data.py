import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import helper
from tensorflow.keras.utils import to_categorical


# Function to parse XML annotation for bounding box
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        boxes.append((int(bbox.find('xmin').text),
                      int(bbox.find('ymin').text),
                      int(bbox.find('xmax').text),
                      int(bbox.find('ymax').text)))
    return boxes

# Function to load and crop images based on bounding box
def load_and_crop_image(image_path, box, small_size_counter, padding=50):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)

    # Original image dimensions
    image_shape = tf.shape(image)
    image_height, image_width = image_shape[0], image_shape[1]

    x1, y1, x2, y2 = box

    # Check initial size of the bounding box
    initial_width = x2 - x1
    initial_height = y2 - y1

    if initial_width < 100 or initial_height < 100:
        small_size_counter['count'] += 1
        return None

    # Adjust box coordinates to include additional padding
    x1_padded = tf.maximum(x1 - padding, 0)
    y1_padded = tf.maximum(y1 - padding, 0)
    x2_padded = tf.minimum(x2 + padding, image_width)
    y2_padded = tf.minimum(y2 + padding, image_height)

    # Compute new box dimensions
    box_width = x2_padded - x1_padded
    box_height = y2_padded - y1_padded

    image_cropped = tf.image.crop_to_bounding_box(image, y1_padded, x1_padded, box_height, box_width)
    image_resized = tf.image.resize(image_cropped, (224, 224))

    return image_resized.numpy()  # Return the processed image as a numpy array

# Function to prepare dataset for 'person' class
def prepare_llvip_dataset(base_path, annotation_folder, image_folders):
    images, labels = [], []
    small_size_counter = {'count': 0}

    annotations_path = os.path.join(base_path, annotation_folder)
    annotation_files = os.listdir(annotations_path)
    for index, annotation_file in enumerate(annotation_files):
        filename = annotation_file.replace('.xml', '.jpg')
        annotation_path = os.path.join(annotations_path, annotation_file)
        boxes = parse_annotation(annotation_path)

        for folder in image_folders:
            image_path = os.path.join(base_path, folder, filename)
            if os.path.exists(image_path):
                for box in boxes:
                    cropped_image = load_and_crop_image(image_path, box, small_size_counter)
                    if cropped_image is not None:
                        images.append(cropped_image)
                        labels.append(1)  # Label for 'person'
                break

    print("Number of LLVIP images where size is too small: " + str(small_size_counter.get('count')))
    return np.array(images), np.array(labels)

# Function to prepare dataset for 'non-person' class
def prepare_iiit_dataset(base_path, list_paths, image_folder):
    images, labels = [], []

    for list_path in list_paths:
        with open(os.path.join(base_path, list_path), 'r') as file:
            lines = file.readlines()
            for index, line in enumerate(lines):
                image_name, _, _, _ = line.split()
                image_path = os.path.join(base_path, image_folder, image_name + '.jpg')
                try:
                    image_raw = tf.io.read_file(image_path)  # Read the raw image file
                    image = tf.image.decode_jpeg(image_raw, channels=1)  # Decode JPEG image
                    image = tf.image.resize(image, (224, 224))  # Resize for model compatibility
                    images.append(image.numpy())  # Convert to numpy array for model compatibility
                    labels.append(0)  # Label for 'non-person'
                except tf.errors.NotFoundError:
                    print(f"The file at {image_path} does not exist.")
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")

    return np.array(images), np.array(labels)

def prepare_flir_dataset(folders):
    images_features = []
    labels = []
    small_size_counter = {'count': 0}

    for folder in folders:
        index_file_path = os.path.join(folder, 'index.json')
        with open(index_file_path, 'r') as file:
            index_data = json.load(file)

        for frame in index_data['frames']:
            image_path = os.path.join(folder, 'data', 'video-' + frame['videoMetadata']['videoId'] + '-frame-' + str(frame['videoMetadata']['frameIndex']).zfill(6) + '-' + frame['datasetFrameId'] + '.jpg')
            for annotation in frame['annotations']:
                if annotation['labels'][0] == 'car':
                    bounding_box = annotation['boundingBox']
                    box = [bounding_box['x'], bounding_box['y'], bounding_box['x'] + bounding_box['w'], bounding_box['y'] + bounding_box['h']]
                    image_feature = load_and_crop_image(image_path, box, small_size_counter)
                    if image_feature is not None:
                        images_features.append(image_feature)
                        labels.append(0)  # All features belong to class '0'

    print("Number of FLIR images where size is too small: " + str(small_size_counter.get('count')))
    return images_features, labels

# Example usage
with tf.device('/cpu:0'):

    # LLVIP data labelled as 'person'
    base_path_llvip = 'raw_data/llvip/raw_data'
    image_folders_llvip = ['grayscale/train', 'grayscale/test']
    annotation_folder_llvip = 'Annotations'
    llvip_images, llvip_labels = prepare_llvip_dataset(base_path_llvip, annotation_folder_llvip, image_folders_llvip)
    print('LLVIP images: ' + str(len(llvip_images)))

    # Oxford IIIT data (cats and pets) labelled as 'non-person'
    base_path_iiit = 'raw_data/oxford-iiit-pet/raw_data'
    list_paths_iiit = ['annotations/annotations/trainval.txt', 'annotations/annotations/test.txt']
    image_folder_iiit = 'images/grayscale'
    iiit_images, iiit_labels = prepare_iiit_dataset(base_path_iiit, list_paths_iiit, image_folder_iiit)
    print('IIIT images: ' + str(len(iiit_images)))

    # FLIR data (cars) labelled as 'non-person'
    folders = ['raw_data/flir/raw_data/images_thermal_train', 'raw_data/flir/raw_data/images_thermal_val']
    flir_images, flir_labels = prepare_flir_dataset(folders)
    print('FLIR images: ' + str(len(flir_images)))

    # Function to randomly sample images and labels from a dataset
    def sample_images_and_labels(images, labels, sample_size):
        # Generate random indices for sampling
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)

        # Select the specified number of random samples
        sampled_indices = indices[:sample_size]
        return images[sampled_indices], labels[sampled_indices]

    # Sample images from each dataset to ensure balanced classes (same amount of images for each class)
    sample_size = len(flir_images) # it's the smallest dataset
    images_llvip_sample, labels_llvip_sample = sample_images_and_labels(llvip_images, llvip_labels, sample_size)
    images_iiit_sample, labels_iiit_sample = sample_images_and_labels(iiit_images, iiit_labels, sample_size)
    images_flir_sample, labels_flir_sample = flir_images, flir_labels # use all flir images
    print("Sampled LLVIP images: " + str(len(images_llvip_sample)))
    print("Sampled IIIT images: " + str(len(images_iiit_sample)))
    print("Sampled FLIR images: " + str(len(images_flir_sample)))

    # Combine the sampled data
    images_combined = np.concatenate((images_llvip_sample, images_iiit_sample, images_flir_sample))
    labels_combined = np.concatenate((labels_llvip_sample, labels_iiit_sample, labels_flir_sample))

    # Shuffle the combined dataset
    indices = np.arange(images_combined.shape[0])
    np.random.shuffle(indices)
    images_shuffled = images_combined[indices]
    labels_shuffled = labels_combined[indices]

    # Split the data into training, validation and test sets
    train_ratio = 0.60
    validation_ratio = 0.2
    test_ratio = 0.2

    # First, split to get the training and the temp (validation+test) sets
    X_train, images_temp, y_train, labels_temp = train_test_split(
        images_shuffled, labels_shuffled, test_size=(1 - train_ratio), stratify=labels_shuffled
    )

    # Then split the temp set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        images_temp, labels_temp, test_size=test_ratio/(test_ratio + validation_ratio), stratify=labels_temp
    )

    # Convert labels to categorical for model training
    y_train_one_hot_encoded = to_categorical(y_train, num_classes=2)
    y_val_one_hot_encoded = to_categorical(y_val, num_classes=2)
    y_test_one_hot_encoded = to_categorical(y_test, num_classes=2)

    # At this point, X_train, X_val, X_test, y_train, y_val, and y_test are ready for training, validating, and testing your model.
    print("Training images: " + str(len(X_train)))
    print("Validation images: " + str(len(X_val)))
    print("Test images: " + str(len(X_test)))

    # File paths
    train_images_file = helper.train_images_file
    val_images_file = helper.val_images_file
    test_images_file = helper.test_images_file
    train_labels_file = helper.train_labels_file
    val_labels_file = helper.val_labels_file
    test_labels_file = helper.test_labels_file
    train_labels_file_one_hot_encoded = helper.train_labels_file_one_hot_encoded
    val_labels_file_one_hot_encoded = helper.val_labels_file_one_hot_encoded
    test_labels_file_one_hot_encoded = helper.test_labels_file_one_hot_encoded

    for file in [train_images_file, val_images_file, test_images_file, train_labels_file, val_labels_file, test_labels_file, train_labels_file_one_hot_encoded, val_labels_file_one_hot_encoded, test_labels_file_one_hot_encoded]:
        if os.path.exists(file):
            os.remove(file)

    np.save(train_images_file, X_train)
    np.save(val_images_file, X_val)
    np.save(test_images_file, X_test)
    with open(train_labels_file, 'wb') as f:
        pickle.dump(y_train, f)
    with open(val_labels_file, 'wb') as f:
        pickle.dump(y_val, f)
    with open(test_labels_file, 'wb') as f:
        pickle.dump(y_test, f)
    with open(train_labels_file_one_hot_encoded, 'wb') as f:
        pickle.dump(y_train_one_hot_encoded, f)
    with open(val_labels_file_one_hot_encoded, 'wb') as f:
        pickle.dump(y_val_one_hot_encoded, f)
    with open(test_labels_file_one_hot_encoded, 'wb') as f:
        pickle.dump(y_test_one_hot_encoded, f)