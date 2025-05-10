# backend/app/api/image.py
import requests
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

router = APIRouter()

# Imgur API endpoint and client ID from environment variables
IMGUR_CLIENT_ID = "6352b06a8a2c101"
IMGUR_API_URL = "https://api.imgur.com/3/image"


@router.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    # Ensure the image is uploaded
    if image is None:
        return JSONResponse(status_code=400, content={"message": "No image file uploaded"})

    # Read image content
    image_data = await image.read()

    # Prepare the form data to upload to Imgur
    files = {
        "image": ("image.jpg", image_data, image.content_type)
    }
    headers = {
        "Authorization": f"Client-ID {IMGUR_CLIENT_ID}"
    }

    # Send POST request to Imgur API
    response = requests.post(IMGUR_API_URL, files=files, headers=headers)

    if response.status_code == 200:
        img_url = response.json().get("data", {}).get("link")
        return {"image_url": img_url}
    else:
        return JSONResponse(status_code=response.status_code, content={"message": "Failed to upload image"})
