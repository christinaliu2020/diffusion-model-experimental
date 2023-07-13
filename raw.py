import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Subset
import matplotlib.pyplot as plt
# Set random seed for reproducibility
torch.manual_seed(42)

# Define the transforms to apply to the data
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,)),
#     transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
# ])

transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
])

# Load the raw MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# Plot some images from the noisy dataset
num_images = 5

# Get random samples from the noisy dataset
sample_indices = torch.randperm(len(train_dataset))[:num_images]
samples = [train_dataset[i][0] for i in sample_indices]

# Create a grid of images
grid = torchvision.utils.make_grid(samples, nrow=num_images)

# Convert the grid tensor to a numpy array and transpose the dimensions
grid_np = grid.numpy().transpose((1, 2, 0))

# Plot the grid of images
plt.imshow(grid_np)
plt.axis('off')
plt.show()


# Split the dataset into train and test sets
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

# Define the data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the ResNet model
resnet = torchvision.models.resnet34(pretrained=False)
num_classes = 10  # Number of output classes for MNIST
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Training loop
num_epochs = 10
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
