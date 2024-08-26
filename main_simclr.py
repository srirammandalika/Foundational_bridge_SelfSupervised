import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from config import DATA_DIR, BATCH_SIZE, DEVICE, NUM_EPOCHS
from models.simclr import SimCLR
from models.main_task_net import MainTaskNet
from datasets.main_task_dataset import MainTaskDataset
from datasets.inpainting_dataset import InpaintingDataset
from utils.train_utils import train_simclr_model, train_model
from utils.eval_utils import evaluate_model
from utils.train_utils import train_simclr_model
import torch
from models.simclr import SimCLR

DEVICE = 'mps' if torch.cuda.is_available() else 'cpu'

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
    weights = ResNet18_Weights.IMAGENET1K_V1
    resnet_base = resnet18(weights=weights)
    
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

    pretext_train_loader = DataLoader(InpaintingDataset(pretext_dataset, transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(MainTaskDataset(*zip(*train_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MainTaskDataset(*zip(*test_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=False)
    full_train_loader = DataLoader(MainTaskDataset(*zip(*full_train_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)

    model_a = SimCLR(base_encoder='resnet18', out_dim=128).to(DEVICE)
    model_a = train_simclr_model(pretext_train_loader, model_a, num_epochs=NUM_EPOCHS, device=DEVICE)
    print("Starting data loading...")
    pretext_train_loader = DataLoader(...)  # Your DataLoader setup
    print("Data loaded successfully.")

    print("Initializing model...")
    model_a = SimCLR(...)  # Your model initialization
    print("Model initialized successfully.")

    print("Starting training...")
    model_a = train_simclr_model(...)  # Your training call
    print("Training complete.")



    model_b = models.resnet18(pretrained=False)
    model_b.fc = torch.nn.Linear(model_b.fc.in_features, len(np.unique(combined_labels)))
    model_b = model_b.to(DEVICE)
    model_b.load_state_dict(model_a.state_dict(), strict=False)
    model_b = train_model(model_b, train_loader, num_epochs=NUM_EPOCHS, device=DEVICE)

    model_b2 = models.resnet18(pretrained=False)
    model_b2.fc = torch.nn.Linear(model_b2.fc.in_features, len(np.unique(combined_labels)))
    model_b2 = model_b2.to(DEVICE)
    model_b2 = train_model(model_b2, train_loader, num_epochs=NUM_EPOCHS, device=DEVICE)

    model_b3 = models.resnet18(pretrained=False)
    model_b3.fc = torch.nn.Linear(model_b3.fc.in_features, len(np.unique(combined_labels)))
    model_b3 = model_b3.to(DEVICE)
    model_b3 = train_model(model_b3, full_train_loader, num_epochs=NUM_EPOCHS, device=DEVICE)

    print('==> Evaluating Model B (fine-tuned on 25%) on test set...')
    evaluate_model(model_b, test_loader, DEVICE)
    
    print('==> Evaluating Model B2 (trained on 25%) on test set...')
    evaluate_model(model_b2, test_loader, DEVICE)

    print('==> Evaluating Model B3 (trained on 85%) on test set...')
    evaluate_model(model_b3, test_loader, DEVICE)

if __name__ == '__main__':
    main()

