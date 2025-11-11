# Training script for classifying controller device images into four types
# Using PyTorch and MobileNetV2 with data augmentation
# 
# Disclosure: This code is cowritten with AI Tools.
# Author: Matijs Behrens
# Date: 11-11-2025
# Version: 1.0

# Please ensure you have the required libraries installed:
# pip install torch torchvision


__version__ = "1.0.0"
__date__    = "2025-11-11"

# 1. Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights

# 2. Define data directories (assume images organized by class in these folders)
train_dir = "./data/train"    # e.g., data/train/Type1, data/train/Type2, ...
val_dir   = "./data/val"      # optional: separate validation images (if available)

# 3. Define augmentation and normalization transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # random crop and resize
    transforms.RandomRotation(degrees=15),                # slight rotation for orientation variance
    transforms.ColorJitter(brightness=0.3, contrast=0.3), # random lighting changes
    
    # Note: We avoid horizontal flip because device text/QR would not appear mirrored in real images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],           # normalize with ImageNet means
                         [0.229, 0.224, 0.225])
])
# For validation, use only resizing and normalization (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # resize shorter side to 224 and crop center
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 4. Create datasets and data loaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_transforms) if os.path.exists(val_dir) else None
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False) if val_dataset else None

# 5. Initialize MobileNetV2 model with pretrained weights and modify final layer
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
# Freeze feature extractor parameters to avoid overfitting small data
for param in model.features.parameters():
    param.requires_grad = False
# Replace classifier's last linear layer to match 4 classes
num_ftrs = model.classifier[1].in_features  # number of input features to final layer
model.classifier[1] = nn.Linear(num_ftrs, 4)

# 6. Set up loss function and optimizer (only parameters of final layer will update)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Since we froze most layers, only the last layer's parameters have grad=True in optimizer

# 7. Training loop (with basic early stopping or fixed epochs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    # (Optional) Validate on val_dataset if available
    if val_loader:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Val Accuracy = {val_acc:.2f}%")
    else:
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

# 8. Save the trained model (state dict)
torch.save(model.state_dict(), "controller_model_weights.pth")
