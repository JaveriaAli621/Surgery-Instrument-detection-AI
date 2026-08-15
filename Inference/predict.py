import json
import torch
import torch.nn as nn
from PIL import Image

from torchvision import models
from torchvision import transforms

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

with open("Models/classes.json") as f:
    class_names = json.load(f)

num_classes = len(class_names)

model = models.efficientnet_b3()

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        "Models/surgical_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

    print(
        f"Prediction: {class_names[pred.item()]}"
    )

    print(
        f"Confidence: {confidence.item()*100:.2f}%"
    )

    return (
        class_names[pred.item()],
        confidence.item()
    )


if __name__ == "__main__":
    predict("test.jpg")