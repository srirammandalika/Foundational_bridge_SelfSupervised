from models.image_generation_net import ImageGenerationNet
from utils.train_utils import train_image_generation_model
from utils.eval_utils import evaluate_model

def run_image_generation_task(pretext_train_loader, device, img_size):
    model = ImageGenerationNet(in_channels=3, img_size=img_size).to(device)
    model = train_image_generation_model(model, pretext_train_loader, num_epochs=45, device=device, target_size=(img_size, img_size))
    return model
