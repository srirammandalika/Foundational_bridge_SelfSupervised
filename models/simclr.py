# models/simclr.py
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, base_encoder='resnet18', out_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = getattr(models, base_encoder)(pretrained=True)

        # Ensure we have the fc layer accessible
        if hasattr(self.encoder, 'fc'):
            in_features = self.encoder.fc.in_features
        else:
            raise AttributeError(f"The encoder {base_encoder} does not have an 'fc' attribute.")

        # Replace the fc layer with Identity and add projection head
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z
