import os
import sys
import argparse
from scipy.interpolate import interp1d
from copy import deepcopy
import numpy as np
import json
import logging
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
import shutil  # Import shutil for file operations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

#### load the segmentator model
from segmentator import unet_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        bce_loss = self.bce(y_true, y_pred)
        dice_loss_value = dice_loss(y_true, y_pred)
        return self.weights[0] * bce_loss + self.weights[1] * dice_loss_value


def interpolate(roi,length):
    roi = deepcopy(roi)
    points = len(roi['intensity'])
    interpolate = interp1d(np.arange(points), roi['intensity'], kind='linear')
    roi['intensity'] = interpolate(np.arange(length) / (length - 1.) * (points - 1.))
    roi['borders'] = np.array(roi['borders'])
    roi['borders'] = roi['borders'] * (length - 1) // (points - 1)
    return roi


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    return 1 - (numerator + 1) / (denominator + 1)

def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    union = tf.reduce_sum(y_true + y_pred, axis=(1, 2)) - intersection
    return tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))

def reshape_labels(labels, target_shape):
    reshaped_labels = np.zeros(target_shape)
    for i, label in enumerate(labels):
        reshaped_labels[i, :, 0] = label
    return reshaped_labels

def train_segmentator(train_data, val_data, train_labels, val_labels, num_epochs, learning_rate, batch_size):
    input_shape = train_data.shape[1:]  # Extract input shape
    model = unet_model(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=CombinedLoss([0.4, 0.2]),
                  metrics=[iou_metric])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('data/weights/segmentator', save_best_only=True, save_format='tf', monitor='val_iou_metric', mode='max')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    logging.info('Training Segmentator...')
    logging.info(f"Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")
    logging.info(f"Validation data shape: {val_data.shape}, Validation labels shape: {val_labels.shape}")

    train_labels_reshaped = reshape_labels(train_labels, train_data.shape)
    val_labels_reshaped = reshape_labels(val_labels, val_data.shape)

    history = model.fit(train_data, train_labels_reshaped, validation_data=(val_data, val_labels_reshaped),
                        epochs=num_epochs, batch_size=batch_size, callbacks=[checkpoint, lr_reducer])

    model.save('data/weights/segmentator_final', save_format='tf')  # Save model in SavedModel format
    return model

def organize_files_by_label(json_files):
    label2file = {0: [], 1: []}
    for file in json_files:
        with open(file) as json_file:
            roi = json.load(json_file)
            label2file[roi['label']].append(file)
    return label2file

def copy_files_to_directory(files, dir, size):
    os.makedirs(dir, exist_ok=True)
    for _ in range(min(len(files), size)):
        idx = np.random.choice(len(files))
        file = files.pop(idx)
        shutil.copy(file, os.path.join(dir, os.path.basename(file)))

def prepare_directories(dirs):
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

def balanced_copy_files_to_directory(label2file, train_dir, val_dir, test_dir, train_ratio, val_ratio):
    for label, files in label2file.items():
        np.random.shuffle(files)
        total_files = len(files)
        train_size = int(train_ratio * total_files)
        val_size = int(val_ratio * total_files)
        test_size = total_files - train_size - val_size
        copy_files_to_directory(files[:train_size], train_dir, train_size)
        copy_files_to_directory(files[train_size:train_size + val_size], val_dir, val_size)
        copy_files_to_directory(files[train_size + val_size:], test_dir, test_size)

def split_json_data(json_files, train_ratio, val_ratio):
    train_dir, val_dir, test_dir = 'data/train', 'data/val', 'data/test'
    prepare_directories([train_dir, val_dir, test_dir])

    label2file = organize_files_by_label(json_files)

    logging.info('Splitting data into training, validation, and test sets...')
    balanced_copy_files_to_directory(label2file, train_dir, val_dir, test_dir, train_ratio, val_ratio)

    logging.info(f'Data splitting completed. Training set: {len(os.listdir(train_dir))} files, '
                 f'Validation set: {len(os.listdir(val_dir))} files, Test set: {len(os.listdir(test_dir))} files.')

def load_data(path, max_len=None):
    data = []
    labels = []
    files = []
    raw_intensity = []
    for file in os.listdir(path):
        if file.endswith('.json'):
            with open(os.path.join(path, file)) as json_file:
                roi = json.load(json_file)
                roi['intensity'] = np.array(roi['intensity'])
                roi['borders'] = np.array(roi['borders'])
                
                data.append(roi['intensity'])
                labels.append(roi['label'])
               
                files.append(file)
                raw_intensity.append(roi['intensity'])
    
    if max_len is None:
        max_len = max(len(seq) for seq in data)
    
    # Pad sequences
    data = tf.keras.preprocessing.sequence.pad_sequences(data, maxlen=max_len, padding='post')
    
    # Reshape data to have a single feature channel
    data = np.expand_dims(data, axis=-1)
    
    print(f"Data shape after padding and reshaping: {data.shape}")  # Debugging
    
    return np.array(data), np.array(labels), max_len, files, raw_intensity

def build_classifier_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(8, 5, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(16, 5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(32, 5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, 5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, 5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, 5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(256, 5, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(512, 5, padding='same', activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))
    return model

def train_classifier(train_data, train_labels, val_data, val_labels, num_epochs, learning_rate, batch_size):
    input_shape = (train_data.shape[1], train_data.shape[2])
    model = build_classifier_model(input_shape)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    logging.info('Training Classifier...')
    model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size,
              validation_data=(val_data, val_labels), callbacks=[callbacks.ModelCheckpoint('data/weights/classifier.h5', save_best_only=True)])

    return model

def evaluate_model(model, data, labels, dataset_name):
    logging.info(f'Evaluating Classifier on {dataset_name} Data...')
    loss, accuracy = model.evaluate(data, labels, verbose=0)
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)
    mcc = matthews_corrcoef(labels, predicted_labels)
    logging.info(f'{dataset_name} Loss: {loss}, {dataset_name} Accuracy: {accuracy}, {dataset_name} MCC: {mcc}')
    return accuracy, mcc, predicted_labels

def plot_confusion_matrix(labels, predictions, dataset_name, pdf_pages):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {dataset_name} Data')
    pdf_pages.savefig()
    plt.close()

def plot_test_cases(test_data, test_labels, predictions, files, raw_intensity, pdf_pages):
    for i, (data, label, pred, file, raw) in enumerate(zip(test_data, test_labels, predictions, files, raw_intensity)):
        plt.figure(figsize=(10, 4))
        plt.plot(raw, label='Raw Intensity')  # Plot the raw intensity values
        plt.title(f'File: {file}\nTrue Label: {label}, Predicted Label: {pred}')
        plt.legend()
        pdf_pages.savefig()
        plt.close()

def plot_predicted_signal_with_peaks(model, test_data, raw_intensities, threshold=0.5):
    predictions = model.predict(test_data)
    predictions = predictions[:, :, 0]

    with PdfPages('predictions.pdf') as pdf_pages:
        num_samples = len(predictions)
        num_pages = num_samples // 6 + (1 if num_samples % 6 > 0 else 0)

        for page in range(num_pages):
            fig, axs = plt.subplots(3, 2, figsize=(15, 18))
            for i in range(6):
                idx = page * 6 + i
                if idx >= num_samples:
                    break
                ax = axs[i // 2, i % 2]
                raw_intensity = raw_intensities[idx]
                pred = predictions[idx]
                min_length = min(len(raw_intensity), len(pred))
                raw_intensity = raw_intensity[:min_length]
                pred = pred[:min_length]
                ax.plot(raw_intensity, label='Raw Intensity')
                ax.fill_between(np.arange(len(raw_intensity)), 0, 1, where=pred > threshold, color='red', alpha=0.3, transform=ax.get_xaxis_transform(), label='Predicted Peaks')
                ax.legend()
                ax.set_title(f'Sample {idx}')
            pdf_pages.savefig(fig)
            plt.close(fig)

def count_labels(labels, dataset_name):
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    logging.info(f'{dataset_name} label counts: {label_counts}')
    return label_counts

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model using LC-MS data.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training (default: 50).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training (default: 0.001).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64).')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data when splitting (default: 0.7).')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data when splitting (default: 0.15).')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Check if the annotation folder exists
    annotation_folder = 'data/annotations'
    if not os.path.exists(annotation_folder):
        logging.error('Annotation data folder does not exist. Exiting.')
        sys.exit(1)
    
    # Collect JSON files
    json_files = set()
    for root, _, files in os.walk(annotation_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.add(os.path.join(root, file))

    if not json_files:
        logging.error("No JSON files found in the annotation folder.")
        sys.exit(1)

    # Split the JSON data
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    split_json_data(json_files, train_ratio, val_ratio)

    # Load datasets
    train_data, train_labels, max_len, _, _ = load_data('data/train')
    val_data, val_labels, _, _, _ = load_data('data/val', max_len=max_len)
    test_data, test_labels, _, test_files, raw_test_intensity = load_data('data/test', max_len=max_len)  # Load test data with the same max_len
    
    # Count labels
    count_labels(train_labels, 'Training')
    count_labels(val_labels, 'Validation')
    count_labels(test_labels, 'Test')
    
    segmentator = train_segmentator(train_data, val_data, train_labels, val_labels, args.epochs, args.learning_rate, args.batch_size)

    # Predict and plot the results
    plot_predicted_signal_with_peaks(segmentator, test_data, raw_test_intensity, threshold=0.5)  # Adjust threshold as needed

    # Train the Classifier
    classifier = train_classifier(train_data, train_labels, val_data, val_labels, args.epochs, args.learning_rate, args.batch_size)

    # Save final model
    classifier.save('data/weights/classifier_final.h5')

    # Evaluate the model on the training set
    train_accuracy, train_mcc, _ = evaluate_model(classifier, train_data, train_labels, 'Training')

    # Evaluate the model on the validation set
    val_accuracy, val_mcc, _ = evaluate_model(classifier, val_data, val_labels, 'Validation')

    # Evaluate the model on the test set
    test_accuracy, test_mcc, test_predictions = evaluate_model(classifier, test_data, test_labels, 'Test')

    # Print accuracy and MCC
    print('Training Accuracy:', train_accuracy, 'Training MCC:', train_mcc)
    print('Validation Accuracy:', val_accuracy, 'Validation MCC:', val_mcc)
    print('Test Accuracy:', test_accuracy, 'Test MCC:', test_mcc)

    # Create PDF to save all plots
    with PdfPages('data/classifier_plots.pdf') as pdf_pages:
        # Plot confusion matrices
        plot_confusion_matrix(train_labels, classifier.predict(train_data).argmax(axis=1), 'Training', pdf_pages)
        plot_confusion_matrix(val_labels, classifier.predict(val_data).argmax(axis=1), 'Validation', pdf_pages)
        plot_confusion_matrix(test_labels, test_predictions, 'Test', pdf_pages)

        # Plot test cases
        plot_test_cases(test_data, test_labels, test_predictions, test_files, raw_test_intensity, pdf_pages)

if __name__ == '__main__':
    main()
