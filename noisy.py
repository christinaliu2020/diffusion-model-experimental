import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import idx2numpy
import gzip
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.models as models
raw_dataset = MNIST('path_to_data_folder', train=True, download=True, transform=transforms.ToTensor())
noise_std = 0.1


class NoisedMNISTDataset(Dataset):
    def __init__(self, raw_dataset, noise_std):
        self.raw_dataset = raw_dataset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        image, label = self.raw_dataset[idx]

        # Add noise to the image
        noised_image = image + torch.randn_like(image) * self.noise_std

        return noised_image, label, image

noised_dataset = NoisedMNISTDataset(raw_dataset, noise_std)

train_noise_size = int(0.8 * len(noised_dataset))
test_noise_size = len(noised_dataset) - train_noise_size
train_noise_dataset, test_noise_dataset = random_split(noised_dataset, [train_noise_size, test_noise_size])

batch_size = 64
train_loader = DataLoader(train_noise_size, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_noise_size, batch_size=batch_size, shuffle=False)

resnet = models.resnet50(pretrained=False)


# Training loop
num_epochs = 10  # Adjust the number of epochs as needed
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01)

for epoch in range(num_epochs):
    # Training steps
    resnet.train()
    for images, labels, _ in train_loader:
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation steps
    resnet.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = 100 * total_correct / total_samples
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')


# Specify the path to your dataset files
train_images_file = 'path/to/dataset1_train_images.idx3-ubyte'
train_labels_file = 'path/to/dataset1_train_labels.idx3-ubyte'
test_images_file = 'path/to/dataset1_test_images.idx3-ubyte'
test_labels_file = 'path/to/dataset1_test_labels.idx3-ubyte'

# Load training images
with gzip.open(train_images_file, 'rb') as f:
    train_images = idx2numpy.convert_from_file(f)

# Load training labels
with gzip.open(train_labels_file, 'rb') as f:
    train_labels = idx2numpy.convert_from_file(f)

# Load test images
with gzip.open(test_images_file, 'rb') as f:
    test_images = idx2numpy.convert_from_file(f)

# Load test labels
with gzip.open(test_labels_file, 'rb') as f:
    test_labels = idx2numpy.convert_from_file(f)
# Split dataset1
dataset1_train, dataset1_test = train_test_split(dataset1, test_size=0.2, random_state=42)

# Split dataset2
dataset2_train, dataset2_test = train_test_split(dataset2, test_size=0.2, random_state=42)

# Split dataset3
dataset3_train, dataset3_test = train_test_split(dataset3, test_size=0.2, random_state=42)
