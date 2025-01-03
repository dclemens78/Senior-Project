# Adam Boulos
# app.py

''' A model that classifies brain scans in order to detect Alzheimer's disease. The results will be outputted on the website '''

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Models.EfNetMRI import build_model  # Import build_model from EfNetMRI
import torch
import os
import io
from PIL import Image
from torchvision import transforms
import io
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Load trained model
model = build_model()
model.load_state_dict(torch.load(os.path.join(ROOT, 'Models', 'Model-Paths', 'best_efnet_model.pth'), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Endpoint for image upload and model inference
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert grayscale images to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Transform the image for model input
        image = transform(image).unsqueeze(0).to(DEVICE)

        # Run model prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        # Map prediction to label
        labels = {0: "Mild Impairment", 1: "Moderate Impairment", 2: "No Impairment", 3: "Very Mild Impairment"}
        predicted_label = labels[predicted.item()]


        return JSONResponse(content={"prediction": predicted_label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Start the app using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
