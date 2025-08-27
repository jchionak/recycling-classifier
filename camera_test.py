import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# -------------------
# Load trained model
# -------------------
from torchvision import models
import torch.nn as nn

# Load your fine-tuned MobileNetV2
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # binary classification
model.load_state_dict(torch.load("recycling_cnn.pth", map_location="cpu"))  # <- your saved weights
model.eval()

# Class names
class_names = ["Recycling", "Trash"]

# -------------------
# Define preprocessing (match training)
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # same as ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------
# Start camera feed
# -------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to PIL for preprocessing
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)  # add batch dimension

    # Model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)  # convert logits -> probabilities
        conf, preds = torch.max(probs, 1)  # confidence + class index
        label = class_names[preds.item()]
        confidence = conf.item() * 100  # convert to %

    # Show prediction on the frame
    cv2.putText(frame, f"Prediction: {label} Confidence: {confidence}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Recycling Bin AI", frame)

    if confidence >= 93 and label == "Recycling":
        print("Open bin!")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
