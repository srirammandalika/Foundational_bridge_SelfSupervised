# models/image_generation_net.py
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageGenerationDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, image
