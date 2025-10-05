import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Global model instance
trained_model = None

# ✅ Class names (same order as training)
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 
               'Rear Breakage', 'Rear Crushed', 'Rear Normal']


# ✅ Define Model
class CarClassifier_resnet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ✅ Prediction function
def predict(image_path):
    global trained_model

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Load model once
    if trained_model is None:
       trained_model = CarClassifier_resnet(num_classes=len(class_names))
       trained_model.load_state_dict(torch.load(r"C:\Code\DL\damage_detection\training\streamlit_app\model\best_model.pth", map_location='cpu'))
       trained_model.eval()


    # Predict
    with torch.no_grad():
        outputs = trained_model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()] 