print("Starting script...")

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

print("Before preprocessing import...")

from Preprocessing.image_processing import get_dataloaders

print("After preprocessing import...")

print("Imports finished.")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

DATASET_PATH = r"Data\SURGICAL TOOLS\SURGICAL TOOLS"
print("Creating dataloaders...")

train_loader, val_loader, class_names = get_dataloaders(
    DATASET_PATH
)

print("Dataloaders created.")

num_classes = len(class_names)

print("Creating model...")

model = models.efficientnet_b3(
    weights=models.EfficientNet_B3_Weights.DEFAULT
)

print("Model created.")

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier[1].parameters(),
    lr=0.001
)

EPOCHS = 10

print("Starting training loop...")

for epoch in range(EPOCHS):
    print(f"Starting Epoch {epoch + 1}")

    model.train()

    running_loss = 0

    print("Entering batch loop...")

    for images, labels in train_loader:
        print("Batch loaded")

        outputs = model(images)
        print("Forward pass complete")

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        print(f"Loss: {loss.item():.4f}")

        print("Loss calculated")

        loss.backward()

        optimizer.step()
        print("Weights updated")

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {running_loss:.4f}"
    )

os.makedirs("Models", exist_ok=True)

torch.save(
    model.state_dict(),
    "Models/surgical_model.pth"
)

with open(
    "Models/classes.json",
    "w"
) as f:
    json.dump(class_names, f)

print("Model saved successfully.")