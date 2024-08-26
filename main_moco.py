import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from config import DATA_DIR, BATCH_SIZE, DEVICE
from models.byol import BYOL
from datasets.dataset_utils import combine_datasets, MainTaskDataset, ImageGenerationDataset
from utils.train_utils import train_byol_model, evaluate_model

def main():
    # Load datasets
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

    # Dataset transformations
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # DataLoaders
    pretext_train_loader = DataLoader(ImageGenerationDataset(pretext_dataset, transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(MainTaskDataset(*zip(*train_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MainTaskDataset(*zip(*test_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=False)
    full_train_loader = DataLoader(MainTaskDataset(*zip(*full_train_dataset), transform=data_transform), batch_size=BATCH_SIZE, shuffle=True)

    # Initialize BYOL model (Model A)
    resnet_base_a = models.resnet18(pretrained=False)
    model_a = BYOL(resnet_base_a, projection_dim=256, moving_average_decay=0.996).to(DEVICE)

    # Train BYOL model (Model A)
    model_a = train_byol_model(model_a, pretext_train_loader, num_epochs=45, device=DEVICE)

    # Fine-tuning and Evaluation

    # Model B - Fine-tuning on 25% of training data
    resnet_base_b = models.resnet18(pretrained=False)
    resnet_base_b.fc = nn.Linear(resnet_base_b.fc.in_features, len(np.unique(combined_labels)))
    model_b = resnet_base_b.to(DEVICE)
    model_b.load_state_dict(model_a.online_encoder.state_dict(), strict=False)
    print('==> Fine-tuning Model B on 25% training data...')
    train_byol_model(model_b, train_loader, num_epochs=45, device=DEVICE)

    print('==> Evaluating Model B (fine-tuned on 25%) on test set...')
    evaluate_model(model_b, test_loader, DEVICE)

    # Model B2 - Training from scratch on 25% of training data
    resnet_base_b2 = models.resnet18(pretrained=False)
    resnet_base_b2.fc = nn.Linear(resnet_base_b2.fc.in_features, len(np.unique(combined_labels)))
    model_b2 = resnet_base_b2.to(DEVICE)
    print('==> Training Model B2 from scratch on 25% training data...')
    train_byol_model(model_b2, train_loader, num_epochs=45, device=DEVICE)

    print('==> Evaluating Model B2 (trained on 25%) on test set...')
    evaluate_model(model_b2, test_loader, DEVICE)

    # Model B3 - Training from scratch on 85% of training data
    resnet_base_b3 = models.resnet18(pretrained=False)
    resnet_base_b3.fc = nn.Linear(resnet_base_b3.fc.in_features, len(np.unique(combined_labels)))
    model_b3 = resnet_base_b3.to(DEVICE)
    print('==> Training Model B3 from scratch on 85% training data...')
    train_byol_model(model_b3, full_train_loader, num_epochs=45, device=DEVICE)

    print('==> Evaluating Model B3 (trained on 85%) on test set...')
    evaluate_model(model_b3, test_loader, DEVICE)

if __name__ == '__main__':
    main()
