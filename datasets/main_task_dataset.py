# datasets/main_task_dataset.py
#import torch
from torch.utils.data import Dataset

class MainTaskDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label
