import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import scipy.stats as stats
import seaborn as sns

# Data file paths
train_images_cropped_file = 'data/train_images_cropped.npy'
val_images_cropped_file = 'data/val_images_cropped.npy'
test_images_cropped_file = 'data/test_images_cropped.npy'
train_labels_cropped_file = 'data/train_labels_cropped.pkl'
val_labels_cropped_file = 'data/val_labels_cropped.pkl'
test_labels_cropped_file = 'data/test_labels_cropped.pkl'
train_images_full_file = 'data/train_images_full.npy'
val_images_full_file = 'data/val_images_full.npy'
test_images_full_file = 'data/test_images_full.npy'
train_labels_full_file = 'data/train_labels_full.pkl'
val_labels_full_file = 'data/val_labels_full.pkl'
test_labels_full_file = 'data/test_labels_full.pkl'
prediction_images_file_person = 'data/prediction_images_person.npy'
prediction_labels_file_person = 'data/prediction_labels_person.pkl'

def load_data_files_cropped():
    X_train = np.load(train_images_cropped_file)
    X_val = np.load(val_images_cropped_file)
    X_test = np.load(test_images_cropped_file)
    with open(train_labels_cropped_file, 'rb') as f:
        y_train = pickle.load(f)
    with open(val_labels_cropped_file, 'rb') as f:
        y_val = pickle.load(f)
    with open(test_labels_cropped_file, 'rb') as f:
        y_test = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_data_files_full():
    X_train = np.load(train_images_full_file)
    X_val = np.load(val_images_full_file)
    X_test = np.load(test_images_full_file)
    with open(train_labels_full_file, 'rb') as f:
        y_train = pickle.load(f)
    with open(val_labels_full_file, 'rb') as f:
        y_val = pickle.load(f)
    with open(test_labels_full_file, 'rb') as f:
        y_test = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_data_files_prediction():
    X_prediction = np.load(prediction_images_file_person)
    with open(prediction_labels_file_person, 'rb') as f:
        y_prediction = pickle.load(f)
    return X_prediction, y_prediction

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


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


    # Compute confusion matrix
    #title = 'Confusion matrix',
    #cmap = plt.cm.Blues
    #normalize = False
    # Get the true labels
    #true_classes = test_generator.classes
    #predicted_classes = np.round(predictions).astype(int).flatten()  # Assuming binary classification
    #cm = confusion_matrix(true_classes, predicted_classes)
    #classes = list(test_generator.class_indices.keys())

    #class_labels = list(test_generator.class_indices.keys())

    # Print classification report
    #report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    #print(report)

    # Plot confusion matrix
    #plt.figure(figsize=(10, 8))

    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)

    #if normalize:
    #    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    #thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #   plt.text(j, i, cm[i, j],
    #             horizontalalignment="center",
    #            color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')

    #plt.show()

def wilson_confidence_interval(p_hat, n, confidence=0.95):
    z = stats.norm.ppf((1 + confidence) / 2)  # for 95% confidence
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
    lower_bound = center - margin
    upper_bound = center + margin
    print(f"Wilson Confidence Interval: {lower_bound:.4f}, {upper_bound:.4f}")
