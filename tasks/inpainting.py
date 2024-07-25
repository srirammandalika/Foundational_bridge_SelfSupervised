# tasks/inpainting.py
# from models.inpainting_net import InpaintingNet
# from datasets.inpainting_dataset import InpaintingDataset
# from utils.train_utils import train_inpainting_model
# from utils.eval_utils import evaluate_model

# def run_inpainting_task(pretext_train_loader, device):
#     model = InpaintingNet(in_channels=3).to(device)
#     model = train_inpainting_model(model, pretext_train_loader, num_epochs=45, device=device)
#     return model

from models.inpainting_net import InpaintingNet
from utils.train_utils import train_inpainting_model

def run_inpainting_task(pretext_train_loader, device):
    model = InpaintingNet(in_channels=3).to(device)
    model = train_inpainting_model(model, pretext_train_loader, num_epochs=45, device=device)
    return model



