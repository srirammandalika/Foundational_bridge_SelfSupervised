import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MLPHead(nn.Module):
    def __init__(self, in_dim, projection_dim):
        super(MLPHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, projection_dim)
        )

    def forward(self, x):
        return self.layers(x)

class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=256, moving_average_decay=0.996):
        super(BYOL, self).__init__()

        # Create online encoder
        self.online_encoder = nn.Sequential(
            base_encoder,
            MLPHead(base_encoder.fc.in_features, projection_dim)
        )
        base_encoder.fc = nn.Identity()  # remove the original fc layer

        # Create target encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)  # target encoder does not update gradients

        self.moving_average_decay = moving_average_decay

        

    def forward(self, x1, x2):
        online_proj1 = self.online_encoder(x1)
        online_proj2 = self.online_encoder(x2)

        # Calculate loss (e.g., MSE between projections)
        loss = F.mse_loss(online_proj1, online_proj2, reduction='mean') # Ensuring it's scalar
        return loss

    @torch.no_grad()
    def update_moving_average(self):
        # update target encoder weights using an exponential moving average
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = target_params.data * self.moving_average_decay + online_params.data * (1 - self.moving_average_decay)

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)
