import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import cv2

# ---- Load model ----
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes

# Load trained weights
model.load_state_dict(torch.load("recycling_cnn.pth", map_location="cpu"))
model.eval()  # set to evaluation mode

# Optional: load saved class mapping (if you saved it earlier)
try:
    with open("labels.json", "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
except:
    idx_to_class = {0: "recycling", 1: "trash"}  # adjust if your dataset mapping is reversed


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def predict_image(image_path, model, transform, idx_to_class):
    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)  # add batch dimension [1,3,224,224]

    # Inference
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)  # convert logits → probabilities
        pred_idx = torch.argmax(probs, dim=1).item()

    return idx_to_class[pred_idx], probs[0][pred_idx].item()


img_cv = cv2.imread("img.png")
label, confidence = predict_image("img.png", model, transform, idx_to_class)

cv2.putText(img_cv, f"{label} {confidence*100:.1f}%", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

cv2.imshow("Prediction", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
