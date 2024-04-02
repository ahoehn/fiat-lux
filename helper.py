import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# Data file paths
train_images_file = 'data/train_images.npy'
val_images_file = 'data/val_images.npy'
test_images_file = 'data/test_images.npy'
train_labels_file = 'data/train_labels.pkl'
val_labels_file = 'data/val_labels.pkl'
test_labels_file = 'data/test_labels.pkl'

def load_data_files():
    X_train = np.load(train_images_file)
    X_val = np.load(val_images_file)
    X_test = np.load(test_images_file)
    with open(train_labels_file, 'rb') as f:
        y_train = pickle.load(f)
    with open(val_labels_file, 'rb') as f:
        y_val = pickle.load(f)
    with open(test_labels_file, 'rb') as f:
        y_test = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Define a function to print evaluation metrics
def print_metrics(eval_result, metric_names):
    for metric, value in zip(metric_names, eval_result):
        print(f'{metric}: {value}')


# Define a function to plot training history
def plot_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()


def plot_confusion_matrix(test_generator, predictions):
    # Compute confusion matrix
    title = 'Confusion matrix',
    cmap = plt.cm.Blues
    normalize = False
    # Get the true labels
    true_classes = test_generator.classes
    predicted_classes = np.round(predictions).astype(int).flatten()  # Assuming binary classification
    cm = confusion_matrix(true_classes, predicted_classes)
    classes = list(test_generator.class_indices.keys())

    class_labels = list(test_generator.class_indices.keys())

    # Print classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()
