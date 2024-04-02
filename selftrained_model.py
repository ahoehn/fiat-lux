from keras.src.layers import Dropout

import helper
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense, Convolution2D, Activation, Input, BatchNormalization
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
plt.style.use('default')

# Load data files
X_train, X_val, X_test, y_train, y_val, y_test = helper.load_data_files()
print(str(len(X_train)))
print(str(len(X_val)))
print(str(len(X_test)))


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
model.add(Dense(2, activation = 'softmax'))

# compile model and initialize weights
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    batch_size=32,  # Adjust based on your dataset size and memory constraints
                    epochs=10,  # Adjust based on the desired number of training epochs
                    validation_data=(X_val, y_val))

print(history.history)
helper.plot_accuracy(history)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

model.save('self-trained.keras')
