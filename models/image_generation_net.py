#from models.image_generation_net import ImageGenerationNet
#from datasets.image_generation_dataset import ImageGenerationDataset
#from utils.train_utils import train_image_generation_model
#from utils.eval_utils import evaluate_image_generation_model

# def run_image_generation_task(pretext_train_loader, device):
#     model = ImageGenerationNet(in_channels=3, img_size=28).to(device)
#     model = train_image_generation_model(model, pretext_train_loader, num_epochs=45, device=device)
#     return model

# tasks/image_generation.py
import torch.nn as nn

class ImageGenerationNet(nn.Module):
    def __init__(self, in_channels, img_size):
        super(ImageGenerationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
