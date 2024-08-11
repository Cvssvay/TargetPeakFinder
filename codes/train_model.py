import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from utils.dataset import ROIDataset
from utils.training import train_model, CombinedLoss, accuracy, iou
from models.cnn_classifier import Classifier
from models.cnn_segmentator import Segmentator
import logging
from shutil import rmtree, copyfile
import json
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def organize_files_by_label(json_files):
    label2file = {0: [], 1: []}
    for file in json_files:
        with open(file) as json_file:
            roi = json.load(json_file)
            label2file[roi['label']].append(file)
    return label2file

def copy_files_to_directory(files, dir, size):
    selected_files = np.random.choice(files, size, replace=False).tolist()
    for file in selected_files:
        files.remove(file)
        copyfile(file, os.path.join(dir, os.path.basename(file)))

def prepare_directories(dirs):
    for dir in dirs:
        if os.path.exists(dir):
            rmtree(dir)
        os.makedirs(dir)

def print_label_distribution(folder, dataset_name):
    label_counts = {0: 0, 1: 0}
    for file in os.listdir(folder):
        if file.endswith('.json'):
            with open(os.path.join(folder, file)) as json_file:
                roi = json.load(json_file)
                label_counts[roi['label']] += 1
    logging.info(f'{dataset_name} label distribution: {label_counts}')

def split_json_data(json_files, train_ratio, val_ratio):
    train_dir, val_dir, test_dir = '../data/train', '../data/val', '../data/test'
    prepare_directories([train_dir, val_dir, test_dir])

    label2file = organize_files_by_label(json_files)

    for label, files in label2file.items():
        np.random.shuffle(files)
        total_files = len(files)
        train_size = int(train_ratio * total_files)
        val_size = int(val_ratio * total_files)
        test_size = total_files - train_size - val_size

        copy_files_to_directory(files, train_dir, train_size)
        copy_files_to_directory(files, val_dir, val_size)
        copy_files_to_directory(files, test_dir, test_size)

    logging.info(f'Data splitting completed. Training set: {len(os.listdir(train_dir))} files, '
                 f'Validation set: {len(os.listdir(val_dir))} files, Test set: {len(os.listdir(test_dir))} files.')

    print_label_distribution(train_dir, 'Training')
    print_label_distribution(val_dir, 'Validation')
    print_label_distribution(test_dir, 'Test')




def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model using LC-MS data.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training (default: 50).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training (default: 0.001).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default: 64).')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data when splitting (default: 0.7).')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data when splitting (default: 0.15).')
    return parser.parse_args()


def run_training(train_loader, val_loader, model, optimizer, classification_metric, segmenatation_metric,
                 scheduler, label_criterion, integration_criterion, intersection_criterion, accumulation,params):
      
    train_loader = train_loader
    val_loader = val_loader
    model = model 
    optimizer = optimizer
    classification_metric = classification_metric
    segmentation_metric = segmenatation_metric
    scheduler = scheduler
    label_criterion = label_criterion
    integration_criterion = integration_criterion
    intersection_criterion = intersection_criterion
    accumulation = accumulation
         # canvas layout (with 3 subplots)
    figure = plt.figure()
    loss_ax = figure.add_subplot(131)
    loss_ax.set_title('Loss function')
    classification_score_ax = figure.add_subplot(132)
    classification_score_ax.set_title('Classification score')
    segmentation_score_ax = figure.add_subplot(133)
    segmentation_score_ax.set_title('Segmentation score')
    canvas = figure
    number_of_epoch = params['number_of_epochs']
    learning_rate = params['learning_rate']
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    (loss_history,
            train_classification_score_history,
            train_segmentation_score_history,
            val_classification_score_history,
            val_segmentation_score_history)=train_model(model, train_loader, val_loader,optimizer, number_of_epoch, 10,
                classification_metric, segmentation_metric,
                scheduler, label_criterion,integration_criterion, intersection_criterion,
                accumulation, loss_ax,classification_score_ax,segmentation_score_ax,figure, canvas)

    return (loss_history,
            train_classification_score_history,
            train_segmentation_score_history,
            val_classification_score_history,
            val_segmentation_score_history)

def main():
    args = parse_arguments()

    annotation_folder = '../data/annotation'
    if not os.path.exists(annotation_folder):
        logging.error('Annotation data folder does not exist. Exiting.')
        sys.exit(1)

    json_files = set()
    for root, _, files in os.walk(annotation_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.add(os.path.join(root, file))

    if not json_files:
        logging.error("No JSON files found in the annotation folder.")
        sys.exit(1)

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    split_json_data(json_files, train_ratio, val_ratio)

# create data loaders
    
    batch_size = args.batch_size
  
    device=torch.device('cpu')
    train_folder='../data/train'
    val_folder='../data/val'
    test_folder='../data/test'
    
    test_dataset = ROIDataset(path=test_folder, device=device, interpolate=True, length=256, balanced=True)
    train_dataset = ROIDataset(path=train_folder, device=device, interpolate=True, length=256, balanced=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = ROIDataset(path=val_folder, device=device, interpolate=True, length=256, balanced=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # classifier

    classifier = Classifier().to(device)
    optimizer = optim.Adam(params=classifier.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    label_criterion = nn.CrossEntropyLoss()
    params={}
    params['number_of_epochs']= args.epochs
    params['learning_rate']=args.learning_rate

    (loss_history,
            train_classification_score_history,
            train_segmentation_score_history,
            val_classification_score_history,
            val_segmentation_score_history)=run_training(train_loader, val_loader, classifier, optimizer, accuracy, None,
                                                scheduler, label_criterion, None, None, 1,params)
   
    # segmentator
    segmentator = Segmentator().to(device)
    optimizer = optim.Adam(params=segmentator.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    integration_criterion = CombinedLoss([0.4, 0.2])
    intersection_criterion = CombinedLoss([0.1, 2])

    (loss_history2,
            train_classification_score_history2,
            train_segmentation_score_history2,
            val_classification_score_history2,
            val_segmentation_score_history2)=run_training(train_loader, val_loader, segmentator, optimizer, None, iou,
                                                scheduler, None, integration_criterion,
                                                intersection_criterion, 1,params)
    


    # train_dataset = ROIDataset(path='data/train', device=torch.device('cpu'))
    # val_dataset = ROIDataset(path='data/val', device=torch.device('cpu'))

    # train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, args.batch_size)

    # classifier = train_classifier(train_loader, val_loader, args.epochs, args.learning_rate)
    # train_segmentator(train_loader, val_loader, args.epochs, args.learning_rate)

if __name__ == '__main__':
    main()
