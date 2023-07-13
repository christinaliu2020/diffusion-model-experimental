import functools

import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import idx2numpy
import gzip
import torch
from torchvision import datasets
from torchvision.datasets import MNIST
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.models as models
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

torch.manual_seed(42)

def marginal_prob(x, t, eps):
    beta_0=0.1
    beta_1=20
    log_mean_coeff = (
            -0.25 * (t ** 2 - eps ** 2) * (beta_1 - beta_0)
            - 0.5 * (t - eps) * beta_0)
    log_mean_coeff = torch.tensor(log_mean_coeff)  # Convert to a tensor
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return std
# Add Gaussian noise to the dataset



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + std_dev * torch.randn_like(x)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_dataset = MNIST(root='data/', train=True, transform=transform, download=True)

tval = 0.08
std_dev = marginal_prob(mnist_dataset, tval, 0.06)

# Plot some images from the noisy dataset
num_images = 5

# Get random samples from the noisy dataset

sample_indices = torch.randperm(len(mnist_dataset))[:num_images]
samples = [mnist_dataset[i][0] for i in sample_indices]

# Create a grid of images
grid = torchvision.utils.make_grid(samples, nrow=num_images)

# Convert the grid tensor to a numpy array and transpose the dimensions
grid_np = grid.numpy().transpose((1, 2, 0))

# Plot the grid of images
plt.imshow(grid_np)
plt.axis('off')
plt.show()



train_size = int(0.8 * len(mnist_dataset))
test_size = len(mnist_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(mnist_dataset, [train_size, test_size])

# Create data loaders
batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


resnet = models.resnet34(pretrained=False)


# Training loop
num_epochs = 10  # Adjust the number of epochs as needed
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = resnet(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# Evaluation on the test set
resnet.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = 100 * total_correct / total_samples
print(f'Test Accuracy: {accuracy:.2f}%')
