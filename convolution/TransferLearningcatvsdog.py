import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

# check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)


print(os.listdir('/kaggle/input/'))

# transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# load dataset
train_dataset = datasets.ImageFolder(
    root='PetImages',
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root='PetImages',
    transform=test_transform
)

print("classes:", train_dataset.classes)
print("total images:", len(train_dataset))

# split
train_size = int(0.8 * len(train_dataset))
test_size  = len(train_dataset) - train_size

indices    = torch.randperm(len(train_dataset)).tolist()

train_subset = Subset(train_dataset, indices[:train_size])
test_subset  = Subset(test_dataset,  indices[train_size:])

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

print("training images:", len(train_subset))
print("testing images:",  len(test_subset))

# load resnet50
model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# freeze all layers
for param in model.parameters():
    param.requires_grad = False

# change last layer
model.fc = nn.Linear(2048, 2)
model    = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

epochs = 2

for epoch in range(epochs):

    # train
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total        += labels.size(0)
        correct      += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    avg_loss       = running_loss / len(train_loader)

    # test
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}]"
          f" Loss: {avg_loss:.4f}"
          f" Train: {train_accuracy:.2f}%"
          f" Test: {test_accuracy:.2f}%")

# save model
torch.save(model.state_dict(), "resnet_catdog.pth")
print("model saved!")