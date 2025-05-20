import io
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import uvicorn
import os

from project.model.classification.cbam_resnet_ssvae import CBAMResNetSSVAE

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ssvae_model.pth')

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transforms (should match your training transforms)
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Load model
def load_model():
    model = CBAMResNetSSVAE(num_classes=2, hidden_dim=256, latent_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# FastAPI app
app = FastAPI()

class PredictRequest(BaseModel):
    image_url: str

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Download image
        response = requests.get(request.image_url)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    # Preprocess image
    transform = get_transforms()
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        logits = outputs['logits'] if 'logits' in outputs else outputs['classification']
        prob = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(prob, 1)
        confidence = confidence.item() * 100
        pred_class = pred_class.item()

    # Map class index to label (adjust as needed)
    class_names = ['Cancer', 'noCancer']
    result = {
        'predicted_class': class_names[pred_class],
        'confidence': confidence,
        'probabilities': {class_names[i]: float(prob[0][i]) for i in range(len(class_names))}
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
