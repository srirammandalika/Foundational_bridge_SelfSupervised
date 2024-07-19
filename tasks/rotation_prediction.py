# tasks/rotation_prediction.py
from models.rotation_net import RotationNet
from datasets.rotation_dataset import RotationDataset
from utils.train_utils import train_model
from utils.eval_utils import evaluate_model

def run_rotation_task(pretext_train_loader, device):
    model = RotationNet(in_channels=3).to(device)
    model = train_model(model, pretext_train_loader, num_epochs=45, device=device)
    return model
