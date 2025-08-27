import json
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("binary-dataset/train", transform=transform)

with open("labels.json","w") as f:
    json.dump(train_data.class_to_idx, f)