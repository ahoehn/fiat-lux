import os
import pickle
import numpy as np
import tensorflow as tf
import helper


# Function to prepare dataset for 'person' class
def prepare_llvip_dataset(base_path, image_folders, filenames):
    images, labels = [], []

    for folder in image_folders:
        full_path = os.path.join(base_path, folder)
        for file in os.listdir(full_path):
            if file.endswith(".jpg") and file in filenames:
                image_path = os.path.join(full_path, file)
                try:
                    image_raw = tf.io.read_file(image_path)  # Read the raw image file
                    image = tf.image.decode_jpeg(image_raw, channels=1)  # Decode JPEG image
                    image = tf.image.resize(image, (224, 224))  # Resize for model compatibility
                    images.append(image.numpy())  # Convert to numpy array for model compatibility
                    labels.append(1)  # Label for 'person'
                except tf.errors.NotFoundError:
                    print(f"The file at {image_path} does not exist.")
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")

    return np.array(images), np.array(labels)

# Example usage
with tf.device('/cpu:0'):

    # LLVIP data labelled as 'person'
    base_path_llvip = 'raw_data/llvip/raw_data'
    image_folders_llvip = ['grayscale/train', 'grayscale/test']
    filenames = ['190003.jpg','190009.jpg','190287.jpg','200002.jpg','200017.jpg','200026.jpg','210241.jpg',
                 '210249.jpg','210254.jpg','210266.jpg','220007.jpg','220018.jpg','220025.jpg','230036.jpg',
                 '230038.jpg','230086.jpg'
                 ]
    llvip_images, llvip_labels = prepare_llvip_dataset(base_path_llvip, image_folders_llvip, filenames)
    print('LLVIP images: ' + str(len(llvip_images)))


    # File paths
    prediction_images_file_person = helper.prediction_images_file_person
    prediction_labels_file_person = helper.prediction_labels_file_person


    for file in [prediction_images_file_person, prediction_labels_file_person]:
        if os.path.exists(file):
            os.remove(file)

    np.save(prediction_images_file_person, llvip_images)
    with open(prediction_labels_file_person, 'wb') as f:
        pickle.dump(llvip_labels, f)