import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from helper import print_metrics, plot_accuracy, plot_confusion_matrix

# Load the VGG16 model pre-trained on ImageNet data, excluding the top layer
base_model = VGG16(weights='imagenet', include_top=False)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Create a new top layer for the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # New FC layer, random init
predictions = Dense(1, activation='sigmoid')(x)  # New softmax layer
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
    'dataset/test',  # Same directory as training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Binary labels
    subset='validation')  # Set as validation data

# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=5)

# Evaluate the model on the test set
eval_result = model.evaluate(validation_generator)
print_metrics(eval_result, model.metrics_names)
plot_accuracy(history)

# Save the model
model.save('human-or-not.keras')

# Plot confusion matrix
validation_generator.reset()  # Resetting generator to ensure correct order of labels and predictions
predictions = model.predict(validation_generator, verbose=1)

plot_confusion_matrix(validation_generator, predictions)
