import torch
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, num_epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
    return model

def train_inpainting_model(model, train_loader, num_epochs, device):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = torch.nn.functional.interpolate(targets, size=outputs.size()[2:], mode='bilinear', align_corners=True)  # Resize targets to match the outputs
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
    return model

