import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from helper import load_data_files
import numpy as np

from helper import wilson_confidence_interval

# Load the model
model = load_model('results/selftrained.keras')

# Load the data
X_train, X_val, X_test, y_train, y_val, y_test = load_data_files()

# Predict the classes using the trained model
y_pred = model.predict(X_test)

# Convert predictions to class labels (i.e. probabilities to 0 or 1)
y_pred_classes = (y_pred >= 0.5).astype(int).flatten()

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#val_accuracy = history.history['val_accuracy'][-1]
#wilson_confidence_interval(val_accuracy, X_val.size)
