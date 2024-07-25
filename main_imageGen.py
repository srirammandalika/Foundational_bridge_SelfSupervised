import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from config import *
from tasks.image_generation import run_image_generation_task
from models.main_task_net import MainTaskNet
from utils.train_utils import train_model
from utils.eval_utils import evaluate_model
from datasets.main_task_dataset import MainTaskDataset
from datasets.image_generation_dataset import ImageGenerationDataset

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
    retinamnist = np.load(os.path.join(DATA_DIR, 'retinamnist.npz'))
    bloodmnist_data = (bloodmnist['train_images'], bloodmnist['train_labels'].squeeze())
    retinamnist_data = (retinamnist['train_images'], retinamnist['train_labels'].squeeze())
    combined_images, combined_labels = combine_datasets(bloodmnist_data, retinamnist_data)

    total_samples = len(combined_labels)
    pretext_size = int(0.6 * total_samples)
    train_size = int(0.25 * total_samples)
    test_size = total_samples - pretext_size - train_size
    dataset = list(zip(combined_images, combined_labels))
    pretext_dataset, remaining_dataset = random_split(dataset, [pretext_size, total_samples - pretext_size])
    train_dataset, test_dataset = random_split(remaining_dataset, [train_size, test_size])
    full_train_dataset, _ = random_split(dataset, [int(0.85 * total_samples), total_samples - int(0.85 * total_samples)])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    pretext_train_loader = DataLoader(ImageGenerationDataset(pretext_dataset, transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(MainTaskDataset(*zip(*train_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MainTaskDataset(*zip(*test_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=False)
    full_train_loader = DataLoader(MainTaskDataset(*zip(*full_train_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)

    model_a = run_image_generation_task(pretext_train_loader, DEVICE, img_size=28)

    model_b = MainTaskNet(in_channels=3, num_classes=len(np.unique(combined_labels))).to(DEVICE)
    model_b.load_state_dict(model_a.state_dict(), strict=False)
    model_b = train_model(model_b, train_loader, num_epochs=45, device=DEVICE)

    print('==> Evaluating Model B (fine-tuned on 25%) on test set...')
    evaluate_model(model_b, test_loader, DEVICE)

    model_b2 = MainTaskNet(in_channels=3, num_classes=len(np.unique(combined_labels))).to(DEVICE)
    model_b2 = train_model(model_b2, train_loader, num_epochs=45, device=DEVICE)

    model_b3 = MainTaskNet(in_channels=3, num_classes=len(np.unique(combined_labels))).to(DEVICE)
    model_b3 = train_model(model_b3, full_train_loader, num_epochs=45, device=DEVICE)

    print('==> Evaluating Model B2 (trained on 25%) on test set...')
    evaluate_model(model_b2, test_loader, DEVICE)

    print('==> Evaluating Model B3 (trained on 85%) on test set...')
    evaluate_model(model_b3, test_loader, DEVICE)

if __name__ == '__main__':
    main()
