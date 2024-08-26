import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

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

class ImageGenerationDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            x1 = self.transform(image)
            x2 = self.transform(image)
        else:
            x1 = x2 = image

        return (x1, x2), label

