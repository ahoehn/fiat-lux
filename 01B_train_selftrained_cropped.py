from keras.src.layers import Dropout
from helper import print_metrics, plot_accuracy, load_data_files_cropped
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Convolution2D, Activation, Input, BatchNormalization
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
plt.style.use('default')

# Load data files
X_train, X_val, X_test, y_train, y_val, y_test = load_data_files_cropped()

model = Sequential()
model.add(Convolution2D(8,kernel_size=(3,3),padding="same", activation = 'relu',input_shape=(224,224,1)))
model.add(Convolution2D(8,kernel_size=(3,3),padding="same", activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(16,kernel_size=(3,3),padding="same", activation = 'relu'))
model.add(Convolution2D(16,kernel_size=(3,3),padding="same", activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dropout((0.5)))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout((0.5)))
model.add(Dense(1, activation = 'sigmoid'))

# compile model and initialize weights
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    batch_size=32,  # Adjust based on your dataset size and memory constraints
                    epochs=15,  # Adjust based on the desired number of training epochs
                    validation_data=(X_val, y_val))

# Save the model
model.save('results/cropped/selftrained.keras')

# Plot the accuracy
plot_accuracy(history)

# Evaluate the model on the test set
eval_result = model.evaluate(X_test, y_test, verbose=1)
print_metrics(eval_result, model.metrics_names)
