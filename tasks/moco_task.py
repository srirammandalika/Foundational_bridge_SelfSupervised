# tasks/moco_task.py

from models.moco import MoCo
from utils.train_utils import train_moco_model
from torchvision import models

def run_moco_task(train_loader, device, img_size):
    # Initialize MoCo model with a base encoder (e.g., ResNet)
    base_encoder = models.resnet18
    moco_model = MoCo(base_encoder, dim=128, K=65536, m=0.999, T=0.07, device=device)
    moco_model = moco_model.to(device)

    # Train the MoCo model
    train_moco_model(moco_model, train_loader, device, num_epochs=45)

    return moco_model
