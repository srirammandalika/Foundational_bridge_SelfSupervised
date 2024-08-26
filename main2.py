import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from config import *
from tasks.inpainting import run_inpainting_task
from models.main_task_net import MainTaskNet
from utils.train_utils import train_model
from utils.eval_utils import evaluate_model
from datasets.main_task_dataset import MainTaskDataset
from datasets.inpainting_dataset import InpaintingDataset

def combine_datasets(*datasets):
    combined_images = []
    combined_labels = []
    label_offset = 0

    for i, (images, labels) in enumerate(datasets):
        if images.ndim == 3:
            images = np.expand_dims(images, axis=-1)
            images = np.repeat(images, 3, axis=-1)
        combined_images.append(images)
        adjusted_labels = labels + label_offset
        combined_labels.append(adjusted_labels)
        label_offset += len(np.unique(labels))
        print(f'Dataset {i+1} labels:', np.unique(adjusted_labels))
        print(f'Dataset {i+1} shape:', images.shape)

    combined_images = np.concatenate(combined_images, axis=0)
    combined_labels = np.concatenate(combined_labels, axis=0)
    print('\nCombined dataset:')
    print('Labels:', np.unique(combined_labels))
    print('Shape:', combined_images.shape)
    return combined_images, combined_labels

def main():
    bloodmnist = np.load(os.path.join(DATA_DIR, 'bloodmnist.npz'))
    pathmnist = np.load(os.path.join(DATA_DIR, 'pathmnist.npz'))
    bloodmnist_data = (bloodmnist['train_images'], bloodmnist['train_labels'].squeeze())
    pathmnist_data = (pathmnist['train_images'], pathmnist['train_labels'].squeeze())
    combined_images, combined_labels = combine_datasets(bloodmnist_data, pathmnist_data)
    
    total_samples = len(combined_labels)
    pretext_size = int(0.65 * total_samples)
    train_size = int(0.25 * total_samples)
    val_test_size = total_samples - pretext_size - train_size
    
    dataset = list(zip(combined_images, combined_labels))
    pretext_dataset, remaining_dataset = random_split(dataset, [pretext_size, total_samples - pretext_size])
    train_dataset, val_test_dataset = random_split(remaining_dataset, [train_size, val_test_size])
    val_size = int(0.5 * val_test_size)
    test_size = val_test_size - val_size
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    pretext_train_loader = DataLoader(InpaintingDataset(pretext_dataset, transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(MainTaskDataset(*zip(*train_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MainTaskDataset(*zip(*val_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(MainTaskDataset(*zip(*test_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=False)

    # Run 1: Train model_a on the pretext dataset
    model_a = run_inpainting_task(pretext_train_loader, DEVICE)

    # Run 2: Fine-tune model_b (from model_a) on the train dataset (25%)
    model_b = MainTaskNet(in_channels=3, num_classes=len(np.unique(combined_labels))).to(DEVICE)
    model_b.load_state_dict(model_a.state_dict(), strict=False)
    del model_a  # Delete model_a after transferring weights
    model_b = train_model(model_b, train_loader, num_epochs=NUM_EPOCHS, device=DEVICE)

    # Evaluate Model B on validation and test sets
    print('==> Evaluating Model B (fine-tuned on 25%) on validation set...')
    accuracy_b_val = evaluate_model(model_b, val_loader, DEVICE)
    print(f'Validation Accuracy: {accuracy_b_val:.2f}%')

    print('==> Evaluating Model B (fine-tuned on 25%) on test set...')
    accuracy_b_test = evaluate_model(model_b, test_loader, DEVICE)
    print(f'Test Accuracy: {accuracy_b_test:.2f}%')

    # Run 3: Train model_b2 (Model C) on 25% of the dataset
    model_b2 = MainTaskNet(in_channels=3, num_classes=len(np.unique(combined_labels))).to(DEVICE)
    model_b2 = train_model(model_b2, train_loader, num_epochs=NUM_EPOCHS, device=DEVICE)

    # Evaluate Model B2 on validation and test sets
    print('==> Evaluating Model B2 (trained on 25%) on validation set...')
    accuracy_b2_val = evaluate_model(model_b2, val_loader, DEVICE)
    print(f'Validation Accuracy: {accuracy_b2_val:.2f}%')

    print('==> Evaluating Model B2 (trained on 25%) on test set...')
    accuracy_b2_test = evaluate_model(model_b2, test_loader, DEVICE)
    print(f'Test Accuracy: {accuracy_b2_test:.2f}%')

    # Run 4: Train model_b3 (Model C) on 95% of the dataset
    full_train_dataset, _ = random_split(dataset, [int(0.95 * total_samples), total_samples - int(0.95 * total_samples)])
    full_train_loader = DataLoader(MainTaskDataset(*zip(*full_train_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    
    model_b3 = MainTaskNet(in_channels=3, num_classes=len(np.unique(combined_labels))).to(DEVICE)
    model_b3 = train_model(model_b3, full_train_loader, num_epochs=NUM_EPOCHS, device=DEVICE)

    # Evaluate Model B3 on validation and test sets
    print('==> Evaluating Model B3 (trained on 95%) on validation set...')
    accuracy_b3_val = evaluate_model(model_b3, val_loader, DEVICE)
    print(f'Validation Accuracy: {accuracy_b3_val:.2f}%')

    print('==> Evaluating Model B3 (trained on 95%) on test set...')
    accuracy_b3_test = evaluate_model(model_b3, test_loader, DEVICE)
    print(f'Test Accuracy: {accuracy_b3_test:.2f}%')

if __name__ == '__main__':
    main()
