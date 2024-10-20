# Adam Boulos
#
# app.py

''' A program that connects our Alzheimer's detection model with our website via FastAPI '''

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from PIL import Image
import torch
from torchvision import transforms
import io
from Models.MRI import build_model  
import os
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Load trained model
model = build_model()
model.load_state_dict(torch.load(os.path.join('Models', 'best_model.pth'), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
        predicted_label = labels[predicted.item()]

        return JSONResponse(content={"prediction": predicted_label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Start the app using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
