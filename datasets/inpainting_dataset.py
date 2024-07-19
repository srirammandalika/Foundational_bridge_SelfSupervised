# datasets/inpainting_dataset.py
from torch.utils.data import Dataset
import numpy as np

class InpaintingDataset(Dataset):
    def __init__(self, images, transform=None, mask_size=7):
        self.images = images
        self.transform = transform
        self.mask_size = mask_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, _ = self.images[idx]
        mask = np.ones_like(image)
        i, j = np.random.randint(0, 28 - self.mask_size, 2)
        mask[i:i + self.mask_size, j:j + self.mask_size] = 0
        
        masked_image = image * mask
        
        if self.transform:
            masked_image = self.transform(masked_image)
            image = self.transform(image)

        return masked_image, image

