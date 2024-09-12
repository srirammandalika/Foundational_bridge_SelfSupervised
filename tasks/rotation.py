import torch
import random

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    if angle == 90:
        return torch.rot90(image, 1, [1, 2])
    elif angle == 180:
        return torch.rot90(image, 2, [1, 2])
    elif angle == 270:
        return torch.rot90(image, 3, [1, 2])
    return image

def generate_rotated_batch(batch):
    """Generate a batch of images with random rotations and corresponding labels."""
    rotated_images = []
    rotation_labels = []
    for img in batch:
        angle = random.choice([0, 90, 180, 270])
        rotated_img = rotate_image(img, angle)
        rotated_images.append(rotated_img)
        rotation_labels.append(angle // 90)  # 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
    
    return torch.stack(rotated_images), torch.tensor(rotation_labels)

def run_rotation_task(loader, device):
    """Run the rotation pretext task to train a model to predict rotations."""
    model = RotationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):  # 2 epochs as specified
        running_loss = 0.0
        for i, (inputs, _) in enumerate(loader):
            inputs, labels = generate_rotated_batch(inputs)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}], Loss: {running_loss / len(loader):.4f}')
    
    return model

class RotationNet(torch.nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(128 * 7 * 7, 256)
        self.fc2 = torch.nn.Linear(256, 4)  # 4 classes: 0°, 90°, 180°, 270°

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
