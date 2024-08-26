import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from tasks.simclr_task import run_simclr_task
from utils.losses import compute_contrastive_loss



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


import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

def train_moco_model(model, train_loader, device, num_epochs=200):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    for epoch in range(num_epochs):
        for images, _ in tqdm(train_loader):
            im_q = images.to(device, dtype=torch.float)
            im_k = images.to(device, dtype=torch.float)

            # Compute output
            logits, labels = model(im_q, im_k)

            # Loss
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(all_labels, all_preds))
    return accuracy

import torch
import torch.nn.functional as F
from tqdm import tqdm

# In train_utils.py

def train_byol_model(model, train_loader, num_epochs, device):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data in tqdm(train_loader):
            # Unpack the data
            x1, x2 = data
            # Ensure data is a tensor
            x1 = torch.stack(x1) if isinstance(x1, list) else x1
            x2 = torch.stack(x2) if isinstance(x2, list) else x2

            # Move to device
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()
            loss = model(x1, x2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return model







def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy


def train_simclr_model(loader, model, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in loader:
            optimizer.zero_grad()
            images = torch.cat([images, images], dim=0)
            images = images.to(device)
            features = model(images)
            loss = compute_contrastive_loss(features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(loader)}')
    return model

