import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from helper import print_metrics, plot_accuracy, plot_confusion_matrix

# Load the ResNet50 model pre-trained on ImageNet data, excluding the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top for your specific task
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)  # Example dense layer
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x) # New softmax layer
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',  # This is the target directory
    target_size=(224, 224),  # All images will be resized to 224x224
    batch_size=32,
    class_mode='binary',  # Binary labels
    subset='training')  # Set as training data

validation_generator = train_datagen.flow_from_directory(
    'dataset/train',  # Same directory as training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Binary labels
    subset='validation')  # Set as validation data

test_datagen = ImageDataGenerator(rescale=1./255)

# Test data generator
test_generator = test_datagen.flow_from_directory(
    'dataset/test',  # This is the target directory for the test data
    target_size=(224, 224),  # Images will be resized to 224x224, same as for training and validation
    batch_size=32,  # You can adjust the batch size if needed
    class_mode='binary',  # Assuming binary classification, use 'categorical' for multi-class
    shuffle=False  # It's important not to shuffle test data if you'll be evaluating predictions
)

# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=3)

# Evaluate the model on the test set
eval_result = model.evaluate(validation_generator)
print_metrics(eval_result, model.metrics_names)
plot_accuracy(history)

# Save the model
model.save('human-or-not-resnet.keras')

# Plot confusion matrix
predictions = model.predict(test_generator, verbose=1)

plot_confusion_matrix(test_generator, predictions)
