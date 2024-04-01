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

model.add(Input(shape=(224, 224, 1)))
model.add(Convolution2D(32,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(32,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(32,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('sigmoid'))

# compile model and initialize weights
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    batch_size=32,  # Adjust based on your dataset size and memory constraints
                    epochs=50,  # Adjust based on the desired number of training epochs
                    validation_data=(X_val, y_val))

print(history.history)
helper.plot_accuracy(history)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")