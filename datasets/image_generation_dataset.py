# datasets/image_generation_dataset.py

from torch.utils.data import Dataset

class ImageGenerationDataset(Dataset):
    def __init__(self, data, transform=None):
        self.images = data['train_images']
        self.labels = data['train_labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            im_q = self.transform(image)
            im_k = self.transform(image)

        return (im_q, im_k)
