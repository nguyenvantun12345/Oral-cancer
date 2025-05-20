import io
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from torchvision import transforms

from project.model.classification.cbam_resnet_ssvae import CBAMResNetSSVAE

MODEL_PATH = "./ssvae_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Load model once
model = CBAMResNetSSVAE(num_classes=2, hidden_dim=256, latent_dim=128).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ✅ Reusable local function for image URL prediction
def predict_from_url(image_url: str) -> dict:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    transform = get_transforms()
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)

        if isinstance(outputs, dict):
            logits = outputs.get('class_output')
            if logits is None:
                raise ValueError(f"Model output dict does not contain 'class_output'. Found keys: {list(outputs.keys())}")
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            raise TypeError(f"Unexpected model output type: {type(outputs)}")

        prob = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(prob, 1)
        confidence = confidence.item()
        pred_class = pred_class.item()

    class_names = ['Cancer', 'noCancer']
    return {
        'predicted_class': class_names[pred_class],
        'confidence': confidence,
        'probabilities': {class_names[i]: float(prob[0][i]) for i in range(len(class_names))}
    }

# FastAPI endpoint using the same logic
router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])

class PredictRequest(BaseModel):
    image_url: str

@router.post("/predict")
def predict(request: PredictRequest):
    return predict_from_url(request.image_url)
