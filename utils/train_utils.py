import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F



def train_model(model, train_loader, num_epochs, device, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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

def train_image_generation_model(model, train_loader, num_epochs, device, target_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        for inputs, _ in tqdm(train_loader):
            inputs = inputs.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Resize outputs to match the target size
            outputs = F.interpolate(outputs, size=target_size)
            
            loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
    return model



