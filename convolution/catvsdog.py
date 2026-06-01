import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from PIL import ImageFile

# Allow partially damaged images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------------
# DEVICE
# -----------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------
# TRAIN TRANSFORMS
# -----------------------------------

train_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# -----------------------------------
# TEST TRANSFORMS
# -----------------------------------

test_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# -----------------------------------
# LOAD DATASETS
# -----------------------------------

train_dataset = datasets.ImageFolder(
    root='PetImages',
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root='PetImages',
    transform=test_transform
)

# -----------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size

indices = torch.randperm(len(train_dataset)).tolist()

train_subset =torch.utils.data. Subset(
    train_dataset,
    indices[:train_size]
)

test_subset = torch.utils.data.Subset(
    test_dataset,
    indices[train_size:]
)

# -----------------------------------
# DATALOADERS
# -----------------------------------

train_loader = DataLoader(
    train_subset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_subset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_subset))
print("Testing images:", len(test_subset))

# -----------------------------------
# CNN MODEL
# -----------------------------------

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):

        # Block 1
        x = self.pool(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        # Block 2
        x = self.pool(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        # Block 3
        x = self.pool(
            F.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        # Flatten
        x = x.view(x.size(0), -1)

        # FC 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # FC 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output
        x = self.fc3(x)

        return x

# MODEL
if 'model' in dir():
    del model
    torch.cuda.empty_cache()



# -----------------------------------
# MODEL
# -----------------------------------

model = CNN().to(device)

# -----------------------------------
# LOSS FUNCTION
# -----------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------
# OPTIMIZER
# -----------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# -----------------------------------
# TRAINING
# -----------------------------------

epochs = 20

for epoch in range(epochs):

    # ---------------- TRAIN ----------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # ---------------- TEST ----------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total

    # ---------------- PRINT ----------------

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

# -----------------------------------
# SAVE MODEL
# -----------------------------------

torch.save(model.state_dict(), "catvsdog.pth")

print("Model saved!")