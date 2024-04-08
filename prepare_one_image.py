import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np

full_path = 'raw_data'
file = 'person.jpg'
image_path = os.path.join(full_path, file)

images, labels = [], []
try:
    image_raw = tf.io.read_file(image_path)  # Read the raw image file
    image = tf.image.decode_jpeg(image_raw, channels=3)  # Decode JPEG image
    image = tf.image.resize(image, (224, 224))  # Resize for model compatibility
    images.append(image.numpy())  # Convert to numpy array for model compatibility
    labels.append(0)  # Label for 'non-person'
except tf.errors.NotFoundError:
    print(f"The file at {image_path} does not exist.")
except Exception as e:
    print(f"Error processing file {image_path}: {e}")

i = np.array(images)
l = np.array(labels)

#model = load_model('results/full/selftrained.keras')
model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
#y_pred = model.predict(i)
#print(y_pred)
# Load and preprocess an image
img = load_img('raw_data/person.jpg', target_size=(224, 224))  # Replace with your image path
img_array = img_to_array(img)
img_array_expanded = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_array_expanded)

# Predict the image
predictions = model.predict(img_preprocessed)

# Decode and print the top 5 predictions
top5 = decode_predictions(predictions, top=5)[0]
for class_id, name, score in top5:
    print(f"{name} ({class_id}): {score * 100:.2f}%")