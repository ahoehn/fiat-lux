import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from helper import load_data_files_full, load_data_files_prediction
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from helper import wilson_confidence_interval

# Load the model
model = load_model('results/full/vgg16.keras')

# Load the data
X_train, X_val, X_test, y_train, y_val, y_test = load_data_files_full()
#X_test, y_test = load_data_files_prediction()

# Replicate grayscale images to have three channels to fit VGG model
X_test = np.repeat(X_test, 3, axis=-1)

# Predict the classes using the trained model
y_pred = model.predict(X_test)

# Convert predictions to class labels (i.e. probabilities to 0 or 1)
y_pred_classes = (y_pred >= 0.5).astype(int).flatten()

# Compute the confusion matrix
labels = [0, 1]
cm = confusion_matrix(y_test, y_pred_classes, labels=labels)

# Plot the confusion matrix
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
#val_accuracy = history.history['val_accuracy'][-1]
#wilson_confidence_interval(val_accuracy, X_val.size)
