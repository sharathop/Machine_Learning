
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# -----------------------------------
# Transform
# -----------------------------------

transform = transforms.ToTensor()

# -----------------------------------
# Full Training Dataset
# -----------------------------------

full_train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)

# -----------------------------------
# Split into Train and Validation
# -----------------------------------

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [50000, 10000]
)

# -----------------------------------
# Test Dataset
# -----------------------------------

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transform,
    download=True
)

# -----------------------------------
# DataLoaders
# -----------------------------------

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

# -----------------------------------
# Neural Network
# -----------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)

        )

    def forward(self, x):

        # Flatten image
        x = x.reshape(x.shape[0], -1)

        output = self.model(x)

        return output


# -----------------------------------
# Create Model
# -----------------------------------

model = NeuralNetwork()

print(model)

# -----------------------------------
# Loss Function
# -----------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------
# Training
# -----------------------------------

epochs = 5

for epoch in range(epochs):

    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"Training Loss: {loss.item():.4f}")

    # -----------------------------------
    # Validation
    # -----------------------------------

    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

print("\nTraining Finished!\n")

# -----------------------------------
# Testing
# -----------------------------------

correct = 0
total = 0

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# -----------------------------------
# Predict Single Image
# -----------------------------------

sample_image, sample_label = test_dataset[0]

model.eval()

with torch.no_grad():

    # Add batch dimension and flatten
    image = sample_image.reshape(1, 28 * 28)

    output = model(image)

    _, prediction = torch.max(output, 1)

    print("\nActual Label:", sample_label)
    print("Predicted Label:", prediction.item())

