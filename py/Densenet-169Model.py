import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset, Subset  
from torchvision import models
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import cv2
import numpy as np

# Custom transformation for histogram equalization
class HistogramEqualization:
    def _call_(self, img):
        img = np.array(img)
        if len(img.shape) == 3 and img.shape[2] == 3:  # Color image
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:  # Grayscale image
            img = cv2.equalizeHist(img)
        return Image.fromarray(img)

# Load the dataset with basic transformations
full_dataset = ImageFolder(root='C:/Users/olivia/OneDrive/Desktop/a/data', transform=transforms.ToTensor())
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

# Custom dataset to apply transformations
class CustomDataset(Dataset):
    def _init_(self, dataset, indices, transform=None):
        self.dataset = Subset(dataset, indices)
        self.transform = transform

    def _len_(self):
        return len(self.dataset)

    def _getitem_(self, idx):
        original_img, label = self.dataset[idx]
        transformed_img = self.transform(original_img) if self.transform else original_img
        return (original_img, transformed_img), (label, label)

# Update the transformations
histogram_equalization = HistogramEqualization()

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomRotation(degrees=5),
    histogram_equalization,
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

train_dataset = CustomDataset(full_dataset, train_indices, transform=train_transforms)
val_dataset = CustomDataset(full_dataset, val_indices, transform=val_transforms)

# Create data loaders
batch_size = 45
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model setup
model = models.densenet169(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize lists to monitor loss and accuracy
train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

# Training function with confusion matrix
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for data in train_loader:
            ((inputs_original, inputs_transformed), (labels_original, labels_transformed)) = data
            inputs_original, inputs_transformed = inputs_original.to(device), inputs_transformed.to(device)
            labels_original, labels_transformed = labels_original.to(device), labels_transformed.to(device)

            # Training phase
            inputs = torch.cat((inputs_original, inputs_transformed), dim=0)
            labels = torch.cat((labels_original, labels_transformed), dim=0)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / (2 * len(train_loader.dataset))
        epoch_acc = running_corrects.double() / (2 * len(train_loader.dataset))

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds = []
        all_labels = []

        for data in val_loader:
            (inputs, _), (labels, _) = data
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.data.view(-1).cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

         # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Save the model if validation accuracy has improved
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'finaltesthisbat45ep10last2.pth')

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    print('Finished Training')
    return model

# Train the model
num_epochs = 10
trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs)

# Plot the training and validation loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.show()