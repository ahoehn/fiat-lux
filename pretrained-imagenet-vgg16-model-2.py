import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from helper import print_metrics, plot_accuracy, plot_confusion_matrix, load_data_files, wilson_confidence_interval

# Load the VGG16 model pre-trained on ImageNet data, excluding the top layer
base_model = VGG16(weights='imagenet', include_top=False)
base_model.trainable = False

# Prepare a lambda function to replicate grayscale images across three channels
def replicate_gray_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

# Define the new input shape for grayscale images
input_shape = (224, 224, 1)  # for grayscale images
new_input = Input(shape=input_shape)

# Replicate the grayscale image across three channels to match VGG16 input requirements
x = Lambda(replicate_gray_to_rgb)(new_input)
x = base_model(x, training=False)  # Ensure the base model is in inference mode

# Add custom layers on top
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(2, activation='softmax')(x)  # Single neuron for binary classification

# Create the new model
model = Model(inputs=new_input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the data
X_train, X_val, X_test, y_train, y_val, y_test = load_data_files();
validation_size = X_val.size

history = model.fit(X_train, y_train,
                    batch_size=32,  # Adjust based on your dataset size and memory constraints
                    epochs=2,  # Adjust based on the desired number of training epochs
                    validation_data=(X_val, y_val))

print(history.history)
plot_accuracy(history)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

val_accuracy = history.history['val_accuracy'][-1]
lower_bound, upper_bound = wilson_confidence_interval(val_accuracy, validation_size)
print(f"Wilson Confidence Interval: {lower_bound:.4f}, {upper_bound:.4f}")
