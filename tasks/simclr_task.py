import torch
from utils.losses import compute_contrastive_loss

device = 'mps' if torch.cuda.is_available() else 'cpu'

def run_simclr_task(loader, model, optimizer, num_epochs, device):
    model.train()
    total_loss = 0

    for epoch in range(num_epochs):
        for (images, _) in loader:
            images = torch.cat([images, images], dim=0)
            images = images.to(device)

            optimizer.zero_grad()
            features = model(images)
            loss = compute_contrastive_loss(features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)
