import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

from helper import print_metrics, plot_accuracy, load_data_files_full

# Load the VGG16 model pre-trained on ImageNet data, excluding the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # sigmoid for binary classification

# Define the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the data
X_train, X_val, X_test, y_train, y_val, y_test = load_data_files_full()

# Replicate grayscale images to have three channels to fit VGG model
X_train = np.repeat(X_train, 3, axis=-1)
X_val = np.repeat(X_val, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Train the model
history = model.fit(X_train, y_train,
                    batch_size=32,  # Adjust based on your dataset size and memory constraints
                    epochs=10,  # Adjust based on the desired number of training epochs
                    validation_data=(X_val, y_val))

# Save the model
model.save('results/full/vgg16.keras')

# Plot the accuracy
plot_accuracy(history)

# Evaluate the model on the test set
eval_result = model.evaluate(X_test, y_test, verbose=1)
print_metrics(eval_result, model.metrics_names)